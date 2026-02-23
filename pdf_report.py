from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.units import inch
from datetime import datetime
import io


def generate_resume_pdf(
    predicted_role,
    confidence,
    resume_score,
    skills_found,
    missing_skills,
    top_roles
):
    """
    Generate a resume analysis PDF and return it as bytes.
    """

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    story = []

    # ---------- Title ----------
    story.append(Paragraph("<b>Resume Analysis Report</b>", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(
        Paragraph(
            f"<b>Generated on:</b> {datetime.now().strftime('%d %b %Y, %H:%M')}",
            styles["Normal"]
        )
    )
    story.append(Spacer(1, 0.3 * inch))

    # ---------- Summary ----------
    story.append(Paragraph("<b>Summary</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(f"Predicted Role: <b>{predicted_role}</b>", styles["Normal"]))
    story.append(Paragraph(f"Confidence: <b>{confidence*100:.1f}%</b>", styles["Normal"]))
    story.append(Paragraph(f"Resume Strength Score: <b>{resume_score}/100</b>", styles["Normal"]))
    story.append(Spacer(1, 0.25 * inch))

    # ---------- Top Roles ----------
    story.append(Paragraph("<b>Top Role Suggestions</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    role_items = [
        ListItem(
            Paragraph(f"{role} — {prob*100:.1f}%", styles["Normal"])
        )
        for role, prob in top_roles
    ]
    story.append(ListFlowable(role_items, bulletType="bullet"))
    story.append(Spacer(1, 0.25 * inch))

    # ---------- Skills ----------
    story.append(Paragraph("<b>Skills Found</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    if skills_found:
        skill_items = [
            ListItem(Paragraph(skill, styles["Normal"]))
            for skill in skills_found
        ]
        story.append(ListFlowable(skill_items, bulletType="bullet"))
    else:
        story.append(Paragraph("No core skills detected.", styles["Normal"]))

    story.append(Spacer(1, 0.25 * inch))

    # ---------- Skill Gap ----------
    story.append(Paragraph("<b>Skill Gap Analysis</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    if missing_skills:
        gap_items = [
            ListItem(Paragraph(skill, styles["Normal"]))
            for skill in missing_skills
        ]
        story.append(ListFlowable(gap_items, bulletType="bullet"))
    else:
        story.append(Paragraph("No major skill gaps identified.", styles["Normal"]))

    # ---------- Build PDF ----------
    doc.build(story)
    buffer.seek(0)

    return buffer.getvalue()
