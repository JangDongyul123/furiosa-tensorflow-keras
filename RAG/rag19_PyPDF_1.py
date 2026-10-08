from langchain_community.document_loaders import TextLoader, PyPDFLoader

path = 'c:/study/furiosa-tensorflow-keras/RAG/_data/'
pdf_loader = PyPDFLoader(path+'AttentionIsAllYouNeed.pdf')

pdf_docs = pdf_loader.load()


print(type(pdf_docs))
print(len(pdf_docs))
print(pdf_docs[0])
