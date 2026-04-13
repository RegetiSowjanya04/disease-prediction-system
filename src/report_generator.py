from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import os

def generate_diagnosis_report(patient_name, symptoms, predicted_disease, confidence, top_predictions, recommendations):
    """Generate PDF diagnosis report"""
    
    # Create reports directory if not exists
    if not os.path.exists('../reports'):
        os.makedirs('../reports')
    
    filename = f"../reports/diagnosis_{patient_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        alignment=1,  # Center
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#3498db'),
        spaceAfter=12
    )
    
    content = []
    
    # Title
    content.append(Paragraph("🏥 DISEASE DIAGNOSIS REPORT", title_style))
    content.append(Spacer(1, 0.2*inch))
    
    # Report Info
    content.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}", styles['Normal']))
    content.append(Spacer(1, 0.2*inch))
    
    # Patient Information
    content.append(Paragraph("📋 PATIENT INFORMATION", heading_style))
    patient_data = [
        ["Patient Name:", patient_name],
        ["Report Date:", datetime.now().strftime('%Y-%m-%d')],
        ["Symptoms Count:", str(len(symptoms))]
    ]
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    content.append(patient_table)
    content.append(Spacer(1, 0.2*inch))
    
    # Symptoms
    content.append(Paragraph("🩺 REPORTED SYMPTOMS", heading_style))
    symptoms_text = ", ".join([s.replace('_', ' ').title() for s in symptoms])
    content.append(Paragraph(symptoms_text, styles['Normal']))
    content.append(Spacer(1, 0.2*inch))
    
    # Diagnosis Results
    content.append(Paragraph("🎯 DIAGNOSIS RESULTS", heading_style))
    content.append(Paragraph(f"<b>Primary Diagnosis:</b> {predicted_disease}", styles['Normal']))
    content.append(Paragraph(f"<b>Confidence Level:</b> {confidence:.1f}%", styles['Normal']))
    content.append(Spacer(1, 0.1*inch))
    
    content.append(Paragraph("<b>Top 5 Possible Conditions:</b>", styles['Normal']))
    for i, (disease, prob) in enumerate(top_predictions.items(), 1):
        content.append(Paragraph(f"  {i}. {disease} - {prob:.1f}% confidence", styles['Normal']))
    
    content.append(Spacer(1, 0.2*inch))
    
    # Recommendations
    content.append(Paragraph("💊 RECOMMENDATIONS", heading_style))
    for rec in recommendations:
        content.append(Paragraph(f"• {rec}", styles['Normal']))
    content.append(Spacer(1, 0.2*inch))
    
    # Disclaimer
    content.append(Paragraph("⚠️ DISCLAIMER", heading_style))
    disclaimer = "This is an AI-generated diagnosis report based on reported symptoms. It is for informational purposes only and does not constitute professional medical advice. Please consult a qualified healthcare provider for proper diagnosis and treatment."
    content.append(Paragraph(disclaimer, styles['Normal']))
    
    # Build PDF
    doc.build(content)
    return filename