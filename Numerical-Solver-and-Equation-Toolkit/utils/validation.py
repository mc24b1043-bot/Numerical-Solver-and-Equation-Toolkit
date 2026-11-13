import ast
def parse_vector(text):
 try:
  x=ast.literal_eval(text)
  if not isinstance(x,(list,tuple)) or not x:raise ValueError()
  return [float(v) for v in x]
 except Exception:raise ValueError("Enter a comma-separated numeric list, for example 1, 2, 3.")