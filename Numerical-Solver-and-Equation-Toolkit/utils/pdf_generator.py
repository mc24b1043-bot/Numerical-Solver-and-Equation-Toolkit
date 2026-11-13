from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
def make_report(title,details):
 out=BytesIO();c=canvas.Canvas(out,pagesize=letter);w,h=letter
 c.setFillColorRGB(.04,.10,.18);c.rect(0,h-105,w,105,fill=1,stroke=0)
 c.setFillColorRGB(.2,.85,.95);c.setFont("Helvetica-Bold",19);c.drawString(45,h-55,"Numerical Solver & Equation Toolkit")
 c.setFillColorRGB(1,1,1);c.setFont("Helvetica",11);c.drawString(45,h-78,title)
 y=h-135;c.setFillColorRGB(.1,.15,.23);c.setFont("Helvetica-Bold",12);c.drawString(45,y,"Calculation summary");y-=24
 c.setFont("Helvetica",10)
 for key,val in details.items():
  text=f"{key}: {val}"
  for i in range(0,len(text),92):
   if y<55:c.showPage();y=h-60;c.setFillColorRGB(.1,.15,.23)
   c.drawString(45,y,text[i:i+92]);y-=16
 c.setFont("Helvetica-Oblique",9);c.drawString(45,35,"Generated from user supplied inputs. Numerical approximations may depend on step size.")
 c.save();out.seek(0);return out.getvalue()