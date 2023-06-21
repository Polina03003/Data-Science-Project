import streamlit as st
def main():
    with open(r'/Users/polinamilicyna/Desktop/ВШЭ и РЭШ/4-ый семестр/Наука о данных/Militsyna_Project Militsyna_Project.py', 'r', encoding='utf-8') as file:
        lang = file.read()
        st.code(lang, language='python')
if __name__ == "__main__":
    main()
