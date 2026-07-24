# class FeatureLoader:
#
#     def load(
#         self,
#         specs: list[SensorSpec],
#         start,
#         end,
#     ) -> pd.DataFrame:
#
#         frames = []
#
#         for spec in specs:
#
#             sensor = resolver.resolve(spec.sensor)
#
#             df = influx.series(
#                 measurement=sensor.measurement,
#                 entity_id=sensor.entity_id,
#                 field=sensor.field,
#                 start=start,
#                 end=end,
#                 resample=...,
#             )
#
#             df = df.rename(columns={"value": spec.name})
#
#             if spec.fill == "ffill":
#                 df = df.ffill()
#
#             elif spec.fill == "interpolate":
#                 df = df.interpolate()
#
#             frames.append(df)
#
#         return pd.concat(frames, axis=1)
