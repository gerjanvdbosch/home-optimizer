import logging
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Float, DateTime, select, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from config import Config

Base = declarative_base()


class SolarForecast(Base):
    """Het tabel-model voor alle zonne- en weerdata."""

    __tablename__ = "solar_forecast"

    timestamp = Column(DateTime, primary_key=True, index=True)

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
    timestamp = Column(DateTime, primary_key=True, index=True)

    # De harde metingen
    grid_import = Column(Float)
    grid_export = Column(Float)
    pv_actual = Column(Float)
    wp_actual = Column(Float)

    room_temp = Column(Float)
    dhw_top = Column(Float)
    dhw_bottom = Column(Float)
    supply_temp = Column(Float)
    compressor_freq = Column(Float)
    hvac_mode = Column(Integer)

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
        """Slaat een DataFrame op. Bestaande timestamps worden bijgewerkt (upsert)."""
        if df.empty:
            return

        session: Session = self.SessionLocal()
        try:
            # We zetten de DataFrame om naar een lijst van dictionaries
            valid_columns = {c.key for c in SolarForecast.__table__.columns}
            records = df.to_dict(orient="records")
            for record in records:
                # 'merge' kijkt naar de primary key (timestamp).
                # Bestaat hij al? Dan update. Bestaat hij niet? Dan insert.
                filtered_record = {
                    k: v for k, v in record.items() if k in valid_columns
                }
                obj = SolarForecast(**filtered_record)
                session.merge(obj)

            session.commit()
            self.logger.debug(f"[DB] {len(records)} records opgeslagen/geüpdate")
        except Exception as e:
            session.rollback()
            self.logger.error(f"[DB] Fout bij opslaan forecast: {e}")
        finally:
            session.close()

    def save_measurement(
        self,
        ts: datetime,
        grid_import: float = None,
        grid_export: float = None,
        pv_actual: float = None,
        wp_actual: float = None,
        room_temp: float = None,
        dhw_top: float = None,
        dhw_bottom: float = None,
        supply_temp: float = None,
        compressor_freq: float = None,
        hvac_mode: int = None,
    ):
        """
        Slaat een meetpunt op of werkt het bij.
        Accepteert None waarden (die worden dan overgeslagen bij update).
        """
        session: Session = self.SessionLocal()
        try:
            # 1. Zoek bestaand record
            record = session.query(Measurement).where(Measurement.timestamp == ts).first()

            if not record:
                # Nieuw record aanmaken
                record = Measurement(timestamp=ts)
                session.add(record)

            # 2. Update velden die zijn meegegeven
            if grid_import is not None:
                record.grid_import = grid_import
            if grid_export is not None:
                record.grid_export = grid_export
            if pv_actual is not None:
                record.pv_actual = pv_actual
            if wp_actual is not None:
                record.wp_actual = wp_actual
            if room_temp is not None:
                record.room_temp = room_temp
            if dhw_top is not None:
                record.dhw_top = dhw_top
            if dhw_bottom is not None:
                record.dhw_bottom = dhw_bottom
            if supply_temp is not None:
                record.supply_temp = supply_temp
            if compressor_freq is not None:
                record.compressor_freq = compressor_freq
            if hvac_mode is not None:
                record.hvac_mode = int(hvac_mode)

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

            # 3. Merge (Inner Join)
            # We willen alleen rijen waar we zowel de meting als het weer van hebben
            df_combined = pd.merge(
                df_meas,
                df_fore,
                on="timestamp",
                how="left"
            ).sort_values("timestamp")

            # 4. Bereken 'load_actual' voor gemak in Pandas
            # Formule: Load = Import - Export + PV
            # (Dit is de 'bruto' load inclusief WP, voor base load moet WP er later nog af)
            df_combined["load_actual"] = (
                df_combined["grid_import"] - df_combined["grid_export"] + df_combined["pv_actual"]
            ).clip(lower=0.0)

            return df_combined

        except Exception as e:
            self.logger.error(f"[DB] Fout bij ophalen training data: {e}")
            return pd.DataFrame()