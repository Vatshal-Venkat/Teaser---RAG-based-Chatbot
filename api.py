import google.generativeai as genai

genai.configure(api_key="AIzaSyDWpwA6O7esDp1S--erQvk_P17DMOvMHt8")
models = genai.list_models()
for m in models:
    print(m)
