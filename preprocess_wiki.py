import argparse
import bs4
import os
import re
import spacy
import threading


def divide_in_chunks(alist: list, num: int):
    length = len(alist)
    chunk_size = (length // num)
    for i in range(0, length, chunk_size):
        yield alist[i: i+chunk_size]


def start_with_threads(task,
                       directories: list,
                       output_path: str,
                       num_threads: int):
    directories_chunks = list(divide_in_chunks(directories, num_threads))

    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=task,
                                  args=(directories_chunks[i],
                                        output_path))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def remove_blank_lines(text: str):
    clean_text = re.sub(r"^(?:[\t ]*(?:\r?\n|\r))+",
                        "",
                        text,
                        flags=re.MULTILINE)
    clean_text = re.sub(r"\n$",
                        "",
                        clean_text)
    return clean_text


def format_text(nlp, text: str):
    doc = nlp(text)
    sentence_each_line = ""
    for sent in doc.sents:
        sentence_each_line = sentence_each_line + sent.text + "\n"
    sentence_each_line = remove_blank_lines(sentence_each_line)
    return sentence_each_line


def preprocess_text(nlp, text: str):
    sentence_each_line = ""

    matches = re.finditer(r"<doc(.|\n)+?<\/doc>", text)
    for match in matches:
        match_str = match.group()
        soup = bs4.BeautifulSoup(match_str, "xml")
        tag = soup.find("doc")
        input_text = tag.contents[0]
        input_text = remove_blank_lines(input_text)
        formatted_text = format_text(nlp, input_text)
        sentence_each_line = sentence_each_line + formatted_text + "\n"

    sentence_each_line = remove_blank_lines(sentence_each_line)
    return sentence_each_line


def preprocess(directories: list, output_path: str):
    nlp = spacy.load("en_core_web_sm")

    for directory in directories:
        files = os.scandir(directory.path)
        for fil in files:
            if fil.is_file():
                print(fil.path)

                with open(fil.path, "r") as xml_file:
                    data = xml_file.read()

                sentence_each_line = preprocess_text(nlp, data)

                file_name = "wiki_{}_{}.txt".format(directory.name,
                                                    fil.name)
                path = os.path.join(output_path, file_name)
                with open(path, "w") as txt_file:
                    txt_file.write(sentence_each_line)


def main():
    input_path = "/data/enwiki/text"
    output_path = "/data/enwiki/clean"

    directories = []
    for directory in os.scandir(input_path):
        if directory.is_dir():
            directories.append(directory)

    if args.threading:
        start_with_threads(preprocess,
                           directories,
                           output_path,
                           20)
    else:
        preprocess(directories, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing options")
    parser.add_argument("--threading", action="store_true",
                        help="Use threads")
    args = parser.parse_args()

    main()
