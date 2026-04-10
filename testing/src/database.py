import logging
import pandas as pd
from datetime import datetime, date, timedelta, time
from sqlalchemy import create_engine, Column, Float, DateTime, select, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from config import Config
from utils import safe_float

Base = declarative_base()


class SolarForecast(Base):
    __tablename__ = "solar_forecast"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, unique=True)

    # Voorspelling (Solcast)
    pv_estimate = Column(Float)
    pv_estimate10 = Column(Float)
    pv_estimate90 = Column(Float)

    # Weerdata (OpenMeteo)
    temp = Column(Float)
    cloud = Column(Float)
    wind = Column(Float)
    radiation = Column(Float)
    diffuse = Column(Float)
    tilted = Column(Float)


class Measurement(Base):
    __tablename__ = "measurement"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, unique=True)

    # De harde metingen
    grid_import = Column(Float)
    grid_export = Column(Float)
    pv_actual = Column(Float)

    wp_ufh = Column(Float)
    wp_dhw = Column(Float)
    wp_leg = Column(Float)

    room_temp = Column(Float)
    dhw_top = Column(Float)
    dhw_bottom = Column(Float)
    target_setpoint = Column(Float)
    supply_temp = Column(Float)
    return_temp = Column(Float)
    hvac_mode = Column(Integer)

    shutter_room = Column(Integer)


class Prediction(Base):
    __tablename__ = "prediction"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, unique=True)

    hvac_mode = Column(Integer)

    t_room_pred = Column(Float)
    t_dhw_pred = Column(Float)
    t_out_pred = Column(Float)

    p_solar_pred = Column(Float)
    p_load_pred = Column(Float)
    p_el_ufh_pred = Column(Float)
    p_el_dhw_pred = Column(Float)

    cop_ufh_pred = Column(Float)
    cop_dhw_pred = Column(Float)

    supply_ufh_pred = Column(Float)
    supply_dhw_pred = Column(Float)

    cost_net_pred = Column(Float)


class Database:
    def __init__(self, config: Config):
        self.logger = logging.getLogger(__name__)
        self.database_path = config.database_path

        self.engine = create_engine(
            f"sqlite:///{self.database_path}", connect_args={"check_same_thread": False}
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        Base.metadata.create_all(bind=self.engine)

    def save_forecast(self, df: pd.DataFrame):
        """
        Slaat een DataFrame op. Bestaande records worden slim bijgewerkt:
        alleen waarden die GEEN NaN zijn, worden overschreven in de DB.
        """
        if df.empty:
            return

        session: Session = self.SessionLocal()
        try:
            # Haal de kolomnamen van je database model op (zodat we geen rommel proberen op te slaan)
            valid_columns = {c.key for c in SolarForecast.__table__.columns}

            # Zet DataFrame om naar een lijst van dictionaries (records)
            records = df.to_dict(orient="records")

            for record in records:
                ts = record.get("timestamp")
                if not ts:
                    continue

                # 1. Zoek het record in de database
                obj = (
                    session.query(SolarForecast)
                    .where(SolarForecast.timestamp == ts)
                    .first()
                )

                if not obj:
                    # --- NIEUW RECORD ---
                    # Maak een nieuw object aan, maar filter eerst alle NaN's eruit.
                    # SQL Alchemy vult zelf NULL in voor ontbrekende velden.
                    clean_record = {
                        k: v
                        for k, v in record.items()
                        if pd.notna(v) and k in valid_columns
                    }
                    obj = SolarForecast(**clean_record)
                    session.add(obj)
                else:
                    # --- BESTAAND RECORD ---
                    # Loop dynamisch door de kolommen heen.
                    # Dit vervangt al die losse 'if grid_import is not None' regels.
                    for k, v in record.items():
                        # Check: is het een geldige kolom? Is het geen timestamp? Is de waarde niet NaN?
                        if k in valid_columns and k != "timestamp" and pd.notna(v):
                            setattr(obj, k, v)  # Update de waarde in het object

            session.commit()
            self.logger.debug(f"[DB] {len(records)} forecast records verwerkt.")
        except Exception as e:
            session.rollback()
            self.logger.error(f"[DB] Fout bij opslaan forecast: {e}")
        finally:
            session.close()

    def save_measurement(self, ts: datetime, **kwargs):
        """
        Slaat een meetpunt op of werkt het bij.
        Accepteert variabele argumenten.
        Voorbeeld: save_measurement(ts, grid_import=500.0, room_temp=20.5)
        """
        if ts is None:
            return

        session: Session = self.SessionLocal()
        try:
            # 1. Haal de geldige kolomnamen op uit je Measurement model
            valid_columns = {c.key for c in Measurement.__table__.columns}

            # 2. Zoek bestaand record
            record = (
                session.query(Measurement).where(Measurement.timestamp == ts).first()
            )

            if not record:
                # Nieuw record aanmaken
                record = Measurement(timestamp=ts)
                session.add(record)

            # 3. Dynamisch updaten
            # kwargs bevat alle argumenten die je meegeeft (bv: grid_import=100)
            for key, value in kwargs.items():

                # Check a: Bestaat deze kolom wel in de database?
                if key not in valid_columns:
                    continue

                # Check b: Is de waarde None? (Sla over)
                if value is None:
                    continue

                # Check c: Is de waarde NaN (Not a Number)? (Sla over)
                # Dit is belangrijk omdat float('nan') != None, maar we willen het niet opslaan.
                if isinstance(value, float) and pd.isna(value):
                    continue

                # Als alles goed is -> update de waarde
                setattr(record, key, value)

            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"[DB] Fout bij opslaan meting: {e}")
        finally:
            session.close()

    def get_history(self, cutoff_date: datetime):
        try:
            with self.engine.connect() as conn:
                # 1. Haal metingen (Targets)
                query_meas = (
                    select(Measurement)
                    .where(Measurement.timestamp >= cutoff_date)
                    .order_by(Measurement.timestamp.asc())
                )
                df_meas = pd.read_sql(query_meas, conn)

                # 2. Haal weerdata (Features)
                query_fore = (
                    select(SolarForecast)
                    .where(SolarForecast.timestamp >= cutoff_date)
                    .order_by(SolarForecast.timestamp.asc())
                )
                df_fore = pd.read_sql(query_fore, conn)

            if df_meas.empty or df_fore.empty:
                return pd.DataFrame()

            # Datetime conversie
            df_meas["timestamp"] = pd.to_datetime(df_meas["timestamp"], utc=True)
            df_fore["timestamp"] = pd.to_datetime(df_fore["timestamp"], utc=True)

            df_meas["dhw_temp"] = (df_meas["dhw_top"] + df_meas["dhw_bottom"]) / 2

            df_meas["wp_actual"] = (
                df_meas["wp_ufh"].fillna(0)
                + df_meas["wp_dhw"].fillna(0)
                + df_meas["wp_leg"].fillna(0)
            ).infer_objects(copy=False)

            # 3. Merge (Inner Join)
            # We willen alleen rijen waar we zowel de meting als het weer van hebben
            df_combined = pd.merge(
                df_meas, df_fore, on="timestamp", how="left"
            ).sort_values("timestamp")

            # 4. Bereken 'load_actual' voor gemak in Pandas
            # Formule: Load = Import - Export + PV
            # (Dit is de 'bruto' load inclusief WP, voor base load moet WP er later nog af)
            df_combined["load_actual"] = (
                df_combined["grid_import"]
                - df_combined["grid_export"]
                + df_combined["pv_actual"]
            ).clip(lower=0.0)

            return df_combined

        except Exception as e:
            self.logger.error(f"[DB] Fout bij ophalen training data: {e}")
            return pd.DataFrame()

    def save_prediction(self, plan: list, start_time: datetime):
        if not plan:
            return

        # Starttijd in UTC (zonder tzinfo) voor de database
        t_start_naive = start_time.replace(tzinfo=None)

        self.logger.debug(f"[DB] Opslaan snapshot vanaf {start_time}")

        session = self.SessionLocal()
        try:
            # Filter plan op de tijdstappen vanaf dit uur
            relevant_rows = [
                row for row in plan if row["time"].replace(tzinfo=None) >= t_start_naive
            ]

            if not relevant_rows:
                self.logger.warning(
                    "[DB] Geen toekomstige tijdstappen in plan om op te slaan."
                )
                return

            # Werk bestaande tijdstappen bij, of maak nieuwe aan (upsert)
            for row in relevant_rows:
                ts_naive = row["time"].replace(tzinfo=None)

                record = session.query(Prediction).filter_by(timestamp=ts_naive).first()
                if not record:
                    record = Prediction(timestamp=ts_naive)
                    session.add(record)

                record.hvac_mode = int(row.get("hvac_mode", 0))
                record.t_room_pred = safe_float(row.get("t_room"))
                record.t_dhw_pred = safe_float(row.get("t_dhw"))
                record.t_out_pred = safe_float(row.get("t_out"))
                record.p_solar_pred = safe_float(row.get("p_solar"))
                record.p_load_pred = safe_float(row.get("p_load"))
                record.p_el_ufh_pred = safe_float(row.get("p_el_ufh"))
                record.p_el_dhw_pred = safe_float(row.get("p_el_dhw"))
                record.cop_ufh_pred = safe_float(row.get("cop_ufh"))
                record.cop_dhw_pred = safe_float(row.get("cop_dhw"))
                record.supply_ufh_pred = safe_float(row.get("supply_ufh"))
                record.supply_dhw_pred = safe_float(row.get("supply_dhw"))
                record.cost_net_pred = safe_float(row.get("cost_net"))

            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"[DB] Fout bij opslaan snapshot: {e}")
        finally:
            session.close()

    def get_predictions(self, target_date: date) -> pd.DataFrame:
        t_start = datetime.combine(target_date, time.min)
        t_end = datetime.combine(target_date + timedelta(days=1), time.min)

        try:
            with self.engine.connect() as conn:
                query = (
                    select(Prediction)
                    .where(
                        Prediction.timestamp >= t_start,
                        Prediction.timestamp < t_end,
                    )
                    .order_by(Prediction.timestamp.asc())
                )
                df = pd.read_sql(query, conn)

            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            return df
        except Exception as e:
            self.logger.error(f"[DB] Fout bij ophalen snapshot: {e}")
            return pd.DataFrame()
