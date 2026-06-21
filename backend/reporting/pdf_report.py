"""Professional PDF medical report generation with ReportLab."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.config import PDF_REPORTS_DIR


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "HospTitle",
            parent=base["Heading1"],
            fontSize=18,
            textColor=colors.HexColor("#1a5276"),
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "Sub",
            parent=base["Normal"],
            fontSize=10,
            textColor=colors.grey,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "heading": ParagraphStyle(
            "Sec",
            parent=base["Heading2"],
            fontSize=12,
            textColor=colors.HexColor("#c0392b"),
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle("Body", parent=base["Normal"], fontSize=10, leading=14, alignment=TA_JUSTIFY),
        "small": ParagraphStyle("Small", parent=base["Normal"], fontSize=8, textColor=colors.grey),
        "disclaimer": ParagraphStyle(
            "Disc",
            parent=base["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#7f8c8d"),
            alignment=TA_JUSTIFY,
            backColor=colors.HexColor("#f8f9fa"),
        ),
    }


def generate_pdf_report(
    patient_label: str,
    human_inputs: dict[str, str],
    ensemble: dict,
    shap_contributors: list,
    recommendations: dict,
    ai_explanation: str,
    username: str,
    output_path: Path | None = None,
) -> Path:
    PDF_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_path or PDF_REPORTS_DIR / f"report_{username}_{ts}.pdf"

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    st = _styles()
    story = []

    # Header band
    story.append(Paragraph("HEART DISEASE RISK ASSESSMENT REPORT", st["title"]))
    story.append(Paragraph("AI-Powered Clinical Decision Support System · Educational Use Only", st["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a5276")))
    story.append(Spacer(1, 8))

    meta_data = [
        ["Report ID", patient_label],
        ["Generated (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")],
        ["Prepared for", username],
        ["System", "Ensemble ML (RF + XGBoost + ANN)"],
    ]
    meta_table = Table(meta_data, colWidths=[1.6 * inch, 4.4 * inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ebf5fb")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # Patient inputs
    story.append(Paragraph("1. Patient Clinical Inputs", st["heading"]))
    inp_rows = [["Parameter", "Value"]] + [[k, v] for k, v in human_inputs.items()]
    inp_table = Table(inp_rows, colWidths=[2.5 * inch, 3.5 * inch])
    inp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
    ]))
    story.append(inp_table)

    # Prediction results
    story.append(Paragraph("2. Prediction Results", st["heading"]))
    ep = ensemble.get("ensemble_probability", 0)
    pred_label = "Heart Disease Indicated" if ensemble.get("ensemble_prediction") == 1 else "No Heart Disease Indicated"
    pred_rows = [
        ["Ensemble Prediction", pred_label],
        ["Ensemble Risk Probability", f"{ep * 100:.1f}%"],
        ["Confidence Score", f"{ensemble.get('confidence', 0) * 100:.1f}%"],
        ["Risk Tier", recommendations.get("tier", "N/A")],
        ["Model Agreement", f"{ensemble.get('model_agreement', 0) * 100:.0f}%"],
    ]
    for name, info in ensemble.get("individual", {}).items():
        pred_rows.append([
            f"{name} probability (w={info.get('weight', 0):.2f})",
            f"{info.get('probability', 0) * 100:.1f}%",
        ])
    pt = Table(pred_rows, colWidths=[2.5 * inch, 3.5 * inch])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#fdedec")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
    ]))
    story.append(pt)

    # SHAP
    story.append(Paragraph("3. SHAP-Based Risk Factors", st["heading"]))
    if shap_contributors:
        shap_rows = [["Feature", "SHAP Impact", "Direction"]]
        for item in shap_contributors[:8]:
            direction = "Increases risk" if item["shap"] > 0 else "Decreases risk"
            shap_rows.append([item["feature"], f"{item['shap']:.4f}", direction])
        st_table = Table(shap_rows, colWidths=[2.2 * inch, 1.4 * inch, 2.4 * inch])
        st_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#c0392b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ]))
        story.append(st_table)
    else:
        story.append(Paragraph("SHAP values unavailable for this assessment.", st["body"]))

    # Recommendations
    story.append(Paragraph("4. Clinical Recommendations (Decision Support)", st["heading"]))
    story.append(Paragraph(escape(recommendations.get("summary", "")), st["body"]))
    for action in recommendations.get("actions", []):
        story.append(Paragraph(f"• {escape(str(action))}", st["body"]))
    story.append(Paragraph("<b>Clinical follow-up:</b>", st["body"]))
    for fu in recommendations.get("clinical_follow_up", []):
        story.append(Paragraph(f"• {escape(str(fu))}", st["body"]))
    story.append(Paragraph(f"<b>Urgency:</b> {escape(str(recommendations.get('urgency', 'N/A')))}", st["body"]))

    # AI explanation
    story.append(Paragraph("5. AI-Generated Explanation", st["heading"]))
    safe_ai = escape(ai_explanation).replace("\n", "<br/>") if ai_explanation else "Not generated."
    story.append(Paragraph(safe_ai[:3500], st["body"]))

    # Disclaimer
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Paragraph(
        "<b>MEDICAL DISCLAIMER:</b> This report is generated by an educational AI clinical decision "
        "support demonstration. It is NOT a medical diagnosis, prescription, or substitute for "
        "professional healthcare. All results must be interpreted by qualified clinicians. "
        "In emergency, call your local emergency number immediately.",
        st["disclaimer"],
    ))

    doc.build(story)
    return path
