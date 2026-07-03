from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='docsloader/csvdata.csv')

docs = loader.load()
#this return all the rows as a document

print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)