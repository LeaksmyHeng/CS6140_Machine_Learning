import re

def substitute(current_text: str, word: str, new_word: str) -> str:
    if re.search(r'[^\w\s]', word):
        current_list = current_text.split()
        for index in range(0, len(current_list)):
            if current_list[index] == word:
                current_list[index] = new_word
        substituted_text = ' '.join(current_list)

    else:
        pattern = r'\b{}\b'.format(word)
        print(pattern)
        substituted_text = re.sub(pattern, new_word, current_text)
    return substituted_text

print(substitute('This is rain rain rain.', 'rain', 'snow'))
print(substitute('This is rain. And snow.', 'rain.', 'snow'))
print(substitute('This is rain. And snow.', '', 'snow'))
print(substitute('This is rain. And snow.', 'rain.', 'snow'))
print(substitute('This is a long long sentence. It has some long words.', 'sentence.', '  '))
print(substitute('I am hungry.', '', 'full'))