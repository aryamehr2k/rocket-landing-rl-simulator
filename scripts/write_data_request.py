"""Write the data sheet template, docs/vehicle_data_request.md and the PDF checklist from datasheet_fields.py.

Usage: python scripts/write_data_request.py (needs reportlab); the example sheet is laid out again, values kept.
"""

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from rocketsim.hop.datasheet_docs import BUILD_COMMAND, TEMPLATE_PATH, refreshed_sheet, request_markdown, template_text
from rocketsim.hop.datasheet_fields import GROUPS

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / TEMPLATE_PATH
EXAMPLE = TEMPLATE.parent / "example.yaml"
MARKDOWN = REPO_ROOT / "docs" / "vehicle_data_request.md"
PDF = REPO_ROOT / "docs" / "electric_vehicle_data_request.pdf"
TITLE = "Electric test vehicle: data sheet"
MARGIN = 12 * mm
BOX_SIZE = 3.0 * mm
BOX_LINE_WIDTH = 0.6
BOX_LIFT = 0.5  # points above the line's bottom, centring the box on the text
COLUMN_WIDTHS = (7 * mm, 70 * mm, 40 * mm, 15 * mm, 26 * mm, 28 * mm)  # tick, what, entry, unit, default, value
FONT, BOLD, CODE = "Helvetica", "Helvetica-Bold", "Courier"
BODY_SIZE, CODE_SIZE, HEADING_SIZE, TITLE_SIZE = 7.5, 7.0, 9.5, 14.0
ROW_PADDING = 3.0  # points above and below each row, room to write a value by hand
LEADING = 1.2  # line height as a multiple of the font size
GRID = colors.Color(0.6, 0.6, 0.6)
HEADER_FILL = colors.Color(0.92, 0.92, 0.92)


class TickBox(Flowable):
    """An empty square to tick by hand."""

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        return BOX_SIZE, BODY_SIZE * LEADING  # one text line, so the box lines up with the text beside it

    def draw(self) -> None:
        self.canv.setLineWidth(BOX_LINE_WIDTH)
        self.canv.rect(0, BOX_LIFT, BOX_SIZE, BOX_SIZE)


def style(name: str, font: str = FONT, size: float = BODY_SIZE, space_after: float = 0.0) -> ParagraphStyle:
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=size * LEADING, spaceAfter=space_after)


def write_pdf(path: Path) -> None:
    """The entry list as one table per group, with a box to tick and room for the value, on A4."""
    body, bold = style("body"), style("bold", BOLD)
    story = [
        Paragraph(TITLE, style("title", BOLD, TITLE_SIZE, 4 * mm)),
        Paragraph(
            f"Measure what is marked required; the rest have defaults that work until you measure them. Enter the "
            f"values in a copy of {TEMPLATE_PATH} named after your vehicle and run "
            f"<font name='Courier'>{escape(BUILD_COMMAND)}</font>. Positions are measured from the top (nose) of the "
            f"vehicle downward. Write each measured value in the Value column, then copy it to the entry of the "
            f"same name in the sheet. docs/vehicle_data_request.md and the template say how to measure each one.",
            style("intro", space_after=3 * mm)),
        Paragraph("Filled in by: ______________________________ &nbsp;&nbsp; Date: ______________",
                  style("by", space_after=4 * mm)),
    ]
    code = style("code", CODE, CODE_SIZE)
    header = ["", *(Paragraph(title, bold) for title in ("What", "Entry", "Unit", "Required or default", "Value"))]
    for group in GROUPS:
        rows = [header]
        for entry in group.entries:
            rows.append([TickBox(), Paragraph(entry.what, body), Paragraph(entry.key, code),
                         Paragraph(entry.unit or "-", body),
                         Paragraph(entry.default_text, bold if entry.required else body), ""])
        table = Table(rows, colWidths=COLUMN_WIDTHS, repeatRows=1)
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, GRID),
            ("BACKGROUND", (0, 0), (-1, 0), HEADER_FILL),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("TOPPADDING", (0, 1), (-1, -1), ROW_PADDING),
            ("BOTTOMPADDING", (0, 1), (-1, -1), ROW_PADDING),
        ]))
        heading = Paragraph(f"{group.title} <font name='{CODE}'>({group.key}:)</font>",
                            style("heading", BOLD, HEADING_SIZE, 1.2 * mm))
        story += [KeepTogether([heading, table]), Spacer(1, 2.5 * mm)]
    document = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                                 bottomMargin=MARGIN, title=TITLE, author="", subject="", creator="")
    document.build(story, onFirstPage=page_number, onLaterPages=page_number)


def page_number(canvas: Canvas, document: SimpleDocTemplate) -> None:
    canvas.setFont(FONT, BODY_SIZE)
    canvas.drawRightString(A4[0] - MARGIN, MARGIN / 2, f"page {document.page}")


def main() -> None:
    TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
    TEMPLATE.write_text(template_text())
    EXAMPLE.write_text(refreshed_sheet(EXAMPLE.read_text()))
    MARKDOWN.write_text(request_markdown())
    write_pdf(PDF)
    for path in (TEMPLATE, EXAMPLE, MARKDOWN, PDF):
        print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
