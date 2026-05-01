import random
import json
from datetime import datetime, timedelta

services = ['api-gateway', 'auth-service', 'payment-svc', 'inventory']
levels = ['INFO'] * 14 + ['WARN'] * 4 + ['ERROR'] * 2

events = []
base_time = datetime.now()

for i in range(50):
    level = random.choice(levels)
    status_code = random.choice([200] * 8 + [500] * 2)
    latency = random.randint(10, 300)

    # inject an anomaly cluster around event 35
    if 33 <= i <= 37:
        level = 'ERROR'
        status_code = 500
        latency = random.randint(1500, 2500)

    event = {
        'ts': (base_time + timedelta(seconds=i)).isoformat(),
        'level': level,
        'service': random.choice(services),
        'latency_ms': latency,
        'status_code': status_code
    }
    events.append(event)

with open('sample.jsonl', 'w') as f:
    for e in events:
        f.write(json.dumps(e) + '\n')

print(f"Generated {len(events)} events → logs/sample.jsonl")
print(f"Anomaly cluster injected at events 33–37")