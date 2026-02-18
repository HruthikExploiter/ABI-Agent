"""
scripts/generate_demo_data.py
Generates a realistic supply chain CSV for testing.
Run: python scripts/generate_demo_data.py
"""

import random
from datetime import date, timedelta
from pathlib import Path
import polars as pl


def generate_data(n_rows: int = 5000) -> pl.DataFrame:
    random.seed(42)

    products = [
        ("P001", "Industrial Pump", "Machinery"),
        ("P002", "Steel Bracket", "Metal Parts"),
        ("P003", "Hydraulic Valve", "Machinery"),
        ("P004", "Circuit Board", "Electronics"),
        ("P005", "Conveyor Belt", "Machinery"),
        ("P006", "Sensor Unit", "Electronics"),
        ("P007", "Aluminium Rod", "Metal Parts"),
        ("P008", "Safety Helmet", "Safety"),
        ("P009", "Forklift Battery", "Energy"),
        ("P010", "Packing Foam", "Packaging"),
    ]

    suppliers = [
        ("S001", "TechSupply Co.", "Germany"),
        ("S002", "PacificParts Ltd.", "China"),
        ("S003", "EuroComponents AG", "France"),
        ("S004", "NorthAm Industrial", "USA"),
        ("S005", "AsiaManufact Inc.", "Japan"),
    ]

    warehouses = ["WH-North", "WH-South", "WH-East", "WH-West", "WH-Central"]
    statuses = ["delivered", "shipped", "pending", "returned", "delayed"]
    weights = [0.55, 0.20, 0.12, 0.08, 0.05]
    start = date(2023, 1, 1)

    rows = []
    for i in range(1, n_rows + 1):
        pid, pname, cat = random.choice(products)
        sid, sname, country = random.choice(suppliers)
        warehouse = random.choice(warehouses)
        status = random.choices(statuses, weights=weights)[0]

        order_date = start + timedelta(days=random.randint(0, 730))
        lead_time = random.randint(3, 45)
        ship_date = order_date + timedelta(days=random.randint(1, 10))
        delivery_date = ship_date + timedelta(days=lead_time)

        qty = random.randint(1, 500)
        unit_price = round(random.uniform(5.0, 2500.0), 2)

        rows.append({
            "order_id": f"ORD-{i:05d}",
            "product_id": pid,
            "product_name": pname,
            "category": cat,
            "quantity": qty,
            "unit_price": unit_price,
            "total_revenue": round(qty * unit_price, 2),
            "order_date": order_date.isoformat(),
            "ship_date": ship_date.isoformat(),
            "delivery_date": delivery_date.isoformat(),
            "lead_time_days": lead_time,
            "supplier_id": sid,
            "supplier_name": sname,
            "country": country,
            "warehouse_id": warehouse,
            "inventory_level": random.randint(0, 10000),
            "reorder_point": random.randint(50, 500),
            "status": status,
        })

    return pl.DataFrame(rows)


if __name__ == "__main__":
    path = Path("data/raw/supply_chain_demo.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_data(5000)
    df.write_csv(path)
    print(f"✓ Generated {len(df)} rows → {path}")
    print(f"  Columns: {df.columns}")