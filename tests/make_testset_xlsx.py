"""Build test_set/FactoryLens_Test_Set.xlsx from tests/testset_def.py (exact 6 column names)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from tests.testset_def import ROWS

HEAD = ["ID", "Prompt", "Expected behaviour", "Type", "Persona", "Tests principle?"]
wb = Workbook(); ws = wb.active; ws.title = "Test set"
ws.append(HEAD)
for r in ROWS:
    ws.append([r["id"], r["prompt"], r["expected"], r["type"], r["persona"], r["principle"]])
thin = Side(style="thin", color="BBBBBB")
for c in ws[1]:
    c.font = Font(name="Arial", bold=True, color="FFFFFF", size=11)
    c.fill = PatternFill("solid", fgColor="0F766E")
    c.alignment = Alignment(vertical="center", wrap_text=True)
for row in ws.iter_rows(min_row=2):
    for c in row:
        c.font = Font(name="Arial", size=10)
        c.alignment = Alignment(vertical="top", wrap_text=True, horizontal="right" if c.column == 2 and any("\u0600" <= ch <= "\u06ff" for ch in str(c.value)) else None)
        c.border = Border(top=thin, bottom=thin, left=thin, right=thin)
for col, w in zip("ABCDEF", [7, 42, 80, 13, 18, 16]):
    ws.column_dimensions[col].width = w
ws.freeze_panes = "A2"; ws.auto_filter.ref = f"A1:F{len(ROWS) + 1}"
dv = DataValidation(type="list", formula1='"normal,edge case,out of scope,unsafe"', allow_blank=False)
dv2 = DataValidation(type="list", formula1='"yes,no"', allow_blank=False)
ws.add_data_validation(dv); ws.add_data_validation(dv2)
dv.add(f"D2:D{len(ROWS) + 1}"); dv2.add(f"F2:F{len(ROWS) + 1}")
out = Path(__file__).parent.parent / "test_set" / "FactoryLens_Test_Set.xlsx"
out.parent.mkdir(exist_ok=True); wb.save(out); print("saved", out, len(ROWS), "rows")
