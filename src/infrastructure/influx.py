from influxdb import InfluxDBClient

client = InfluxDBClient(
    host="homeassistant.local",
    port=8086,
    username="home_assistant",
    password="home_assistant",
    database="autogen"
)

result = client.query(
    'SELECT "value" FROM "home_assistant"."autogen"."W" WHERE "entity_id"=\'pv_output\' LIMIT 10'
)

for point in result.get_points():
    print(point)