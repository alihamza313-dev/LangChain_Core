from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """File Upload : Navigate to File Upload lab and upload a PHP web shell. Screenshot the success message. Execute whoami and dir through the shell. Screenshot each result. Now switch to Medium security and try uploading the same shell. What happens and why? Bypass the Medium security filter and successfully upload the shell again. Screenshot 
the bypass method you used."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)
print(chunks[1])