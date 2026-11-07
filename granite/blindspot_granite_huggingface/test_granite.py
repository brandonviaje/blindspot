import json
from granite_api import extract_vulnerability

if __name__ == "__main__":
    description = (
        "An authentication bypass vulnerability in ExampleVPN allows "
        "an unauthenticated remote attacker to access an affected VPN gateway."
    )
    result = extract_vulnerability(description)
    print(json.dumps(result, indent=2))
