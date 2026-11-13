import plotly.express as px
def line(data,x,y,title):
 fig=px.line(data,x=x,y=y,title=title,template="plotly_dark")
 fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=20,r=20,t=55,b=20))
 return fig