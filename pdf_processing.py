import io
import string
from collections import Counter
from typing import List, Optional
from unicodedata import normalize

import fitz # PyMuPDF
import re

END_OF_SENTENCE_CHARACTERS = {'.', '?', '!'}

FOOTER_PATTERN = re.compile(r'Page\s+\d+\s+of\s+\d*', re.IGNORECASE)
LIST_PATTERN = re.compile(r'^[a-zA-Z]\.|^([0-9]*\.[0-9]*)+')

def is_footer(s):
    return re.search(FOOTER_PATTERN, s) 

def is_camel_case(s):
    return s != s.lower() and s != s.upper()

def is_header(s):
    return s.count(".") > 10

def preprocess(s):
    s = re.sub(LIST_PATTERN, "", s)
    return s.strip()


def prefilter_line(line: str):
    return not is_footer(line) and not is_header(line)

def filter_line(line: str):
    return line and len(line.split()) > 3 and len(line.strip()) > 30 \
           and any(char.isalpha() for char in line) and not is_footer(line) and \
           all(char in string.printable for char in line)


def merge_next_line(line_1: str, line_2: str) -> bool:
    """Checks whether line_2 should be concatenated with line_1.
    Assumes that continuation of the next sentence from
    the current line occurs only when this line is full.
    """
    return len(line_1) > 30 and not line_1.isupper() and \
           line_1.strip()[-1] not in END_OF_SENTENCE_CHARACTERS and \
           any(char.isalpha() for char in line_2) and \
           not all(map(is_camel_case, line_1))


def retrieve_lines_from_pdf_file(file_buffer: Optional[io.BytesIO]=None) -> List[str]:
    if not file_buffer:
        return []
    doc = fitz.open(None, file_buffer, filetype="pdf")
    lines = [line.strip() for page in doc for line in page.get_text("text").split("\n") if line.strip()]
    
    repeats_counter = Counter(lines)
    lines = [normalize('NFKC', line) for line in lines if repeats_counter[line] < doc.pageCount / 2] # delete headers

    lines = list(filter(prefilter_line, lines))
    lines = list(map(preprocess, lines))
    concatenated_lines = [lines[0]]
    for i in range(1, len(lines)):
        line_1_text = concatenated_lines[-1]
        line_2_text = lines[i]

        merge = merge_next_line(line_1_text, line_2_text)

        if not merge:
            concatenated_lines.append(line_2_text)
            continue

        if line_1_text[-1] == "-":
            concatenated_lines[-1] = f"{line_1_text[:-1]}{line_2_text}" 
        else:
            concatenated_lines[-1] = f"{line_1_text} {line_2_text}"

    filtered_lines = list(filter(filter_line, concatenated_lines))
    return filtered_lines

if __name__=="__main__":
    # with open("test_file.pdf", "rb") as fin:
    # with open("test_1.pdf", "rb") as fin:
    # with open("0000 - inventory.pdf", "rb") as fin:
    with open("2003 - agentmom.pdf", "rb") as fin:
        with open("file.txt", "w+") as fout:
            f = io.BytesIO(fin.read())
            fout.writelines("\n".join(retrieve_lines_from_pdf_file(f)))