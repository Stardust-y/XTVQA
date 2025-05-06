import fitz
output_path = "../data/pageqa/pdfs/1912.01214.pdf"
doc = fitz.open(output_path)

for i, page in enumerate(doc):
    print(page)
    text_instances = page.search_for("We compare our approaches with related approaches of pivoting")
    print(text_instances)