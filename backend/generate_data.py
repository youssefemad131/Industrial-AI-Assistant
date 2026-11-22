"""Generate the SYNTHETIC factory data used by FactoryLens AI.

Everything here is fictional. Run:  python -m backend.generate_data
Fixed random seed -> the same CSVs every time (so demo numbers are repeatable).
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data"
rng = np.random.default_rng(42)

WEEK_PREV = pd.date_range("2026-10-18", "2026-10-22")  # Sun-Thu
WEEK_LATEST = pd.date_range("2026-10-25", "2026-10-29")
DEFECTS = ["paint run", "orange peel", "uneven coat", "scratch"]


def inspection_records():
    rows, n = [], 0
    for day in list(WEEK_PREV) + list(WEEK_LATEST):
        latest = day in WEEK_LATEST
        for shift in (1, 2, 3):
            for k in range(4):
                n += 1
                if latest and shift == 3:
                    machine = "P2" if k < 3 else "P1"
                else:
                    machine = rng.choice(["P1", "P2"])
                product = rng.choice(["Ceiling fan 56in", "Stand fan 18in"])
                inspected = int(rng.integers(60, 101))
                if not latest:
                    p = 0.021
                elif shift == 3 and machine == "P2":
                    p = 0.075
                elif shift == 3:
                    p = 0.03
                else:
                    p = 0.027
                defects = int(rng.binomial(inspected, p))
                if defects == 0:
                    dtype = "none"
                elif latest and shift == 3:
                    dtype = rng.choice(DEFECTS, p=[0.3, 0.5, 0.2, 0.0])
                else:
                    dtype = rng.choice(DEFECTS)
                lot = "L-014"
                if latest and machine == "P2":
                    lot = "L-018"
                rows.append(dict(record_id=f"INS-{n:03d}", date=day.date().isoformat(),
                                 shift=shift, machine_id=machine, product=product,
                                 batch_id=f"B-{day.strftime('%m%d')}-{shift}{k+1}",
                                 defect_type=dtype, inspected_qty=inspected,
                                 defect_qty=defects, supplier_lot=lot))
    df = pd.DataFrame(rows)
    # deliberate data-quality gaps (so Missing data is real, not decorative)
    gap = df[df.defect_qty > 0].sample(3, random_state=7).index
    df.loc[gap, "defect_type"] = np.nan
    df.loc[df.sample(2, random_state=3).index, "supplier_lot"] = np.nan
    return df


def shift_production():
    rows = []
    for day in list(WEEK_PREV) + list(WEEK_LATEST):
        latest = day in WEEK_LATEST
        for shift in (1, 2, 3):
            down = int(rng.integers(8, 20)) + (18 if latest and shift == 3 else 0)
            rows.append(dict(date=day.date().isoformat(), shift=shift,
                             units_produced=int(rng.integers(480, 561)),
                             downtime_min=down))
    return pd.DataFrame(rows)


def maintenance_events():
    ev = [
        ("2026-10-18", "P1", 1, "lubrication", "completed", "Routine conveyor lubrication."),
        ("2026-10-18", "P2", 2, "belt_check", "completed", "Belt tension checked, within range."),
        ("2026-10-19", "P1", 1, "filter_change", "completed", "Air filter changed."),
        ("2026-10-19", "P2", 3, "lubrication", "completed", "Routine conveyor lubrication."),
        ("2026-10-20", "P1", 2, "compressor_check", "completed", "Compressor pressure normal."),
        ("2026-10-20", "P2", 1, "filter_change", "completed", "Air filter changed."),
        ("2026-10-21", "P1", 1, "nozzle_cleaning", "completed", "Spray gun nozzle cleaned."),
        ("2026-10-21", "P2", 2, "belt_check", "completed", "Belt tension checked, within range."),
        ("2026-10-22", "P1", 3, "lubrication", "completed", "Routine conveyor lubrication."),
        ("2026-10-22", "P2", 1, "compressor_check", "completed", "Compressor pressure normal."),
        ("2026-10-22", "P1", 2, "conveyor_jam", "completed", "Short jam cleared, 10 min stop."),
        ("2026-10-23", "P2", None, "calibration_due", "open", "Oven temperature sensor calibration overdue by 12 days."),
        ("2026-10-23", "P1", 1, "filter_change", "completed", "Air filter changed."),
        ("2026-10-24", "P1", 2, "belt_check", "completed", "Belt tension checked, within range."),
        ("2026-10-25", "P1", 1, "lubrication", "completed", "Routine conveyor lubrication."),
        ("2026-10-25", "P2", 2, "compressor_check", "completed", "Compressor pressure normal."),
        ("2026-10-26", "P2", 3, "temperature_note", "noted", "Operator note: oven temperature reading about 8 C above setpoint."),
        ("2026-10-26", "P1", 1, "nozzle_cleaning", "completed", "Spray gun nozzle cleaned."),
        ("2026-10-27", "P2", 2, "nozzle_replaced", "completed", "Spray gun nozzle replaced."),
        ("2026-10-27", "P1", 3, "filter_change", "completed", "Air filter changed."),
        ("2026-10-28", "P2", 1, "belt_check", "completed", "Belt tension checked, within range."),
        ("2026-10-28", "P1", 2, "lubrication", "completed", "Routine conveyor lubrication."),
        ("2026-10-29", "P2", 3, "conveyor_jam", "completed", "Jam cleared, 25 min stop."),
        ("2026-10-29", "P1", 1, "compressor_check", "completed", "Compressor pressure normal."),
        ("2026-10-29", "P2", 2, "lubrication", "completed", "Routine conveyor lubrication."),
    ]
    df = pd.DataFrame(ev, columns=["date", "machine_id", "shift", "event_type", "status", "notes"])
    df.insert(0, "event_id", [f"MNT-{i+1:03d}" for i in range(len(df))])
    return df


def supplier_lots():
    rows = []
    for i in range(1, 21):
        rows.append(dict(lot_id=f"L-{i:03d}", supplier_code=["SUP-A", "SUP-B", "SUP-C"][i % 3],
                         material=["paint base", "hardener", "thinner"][i % 3],
                         delivery_date=(pd.Timestamp("2026-09-20") + pd.Timedelta(days=int(i * 1.4))).date().isoformat(),
                         inspection_status="passed"))
    df = pd.DataFrame(rows)
    df.loc[df.lot_id == "L-014", ["supplier_code", "material", "delivery_date"]] = ["SUP-A", "paint base", "2026-10-08"]
    df.loc[df.lot_id == "L-018", ["supplier_code", "material", "delivery_date", "inspection_status"]] = \
        ["SUP-B", "paint base", "2026-10-24", "not_inspected"]
    df.loc[df.lot_id == "L-009", "inspection_status"] = "conditional"
    return df


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, df in [("inspection_records", inspection_records()), ("shift_production", shift_production()),
                     ("maintenance_events", maintenance_events()), ("supplier_lots", supplier_lots())]:
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(name, len(df))
