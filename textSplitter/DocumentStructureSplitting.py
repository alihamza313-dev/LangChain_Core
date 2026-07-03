from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """def calculate_area(length, width):
        area = length * width
        return area

    # Main program
    length = float(input("Enter the length: "))
    width = float(input("Enter the width: "))

    result = calculate_area(length, width)

    print(f"The area of the rectangle is: {result}")"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 100,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks[0])
print("--------------------")
print(chunks[1])
print("--------------------")
print(chunks[2])
print("--------------------")
print(chunks[3])