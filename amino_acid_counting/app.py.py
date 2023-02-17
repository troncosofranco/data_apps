# Import libraries

import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image


#=========== Head section======================

st.write("""
# Amino acids Count
This app counts the amino acids composition of query protein chain!
***
""")

#head Image
image = Image.open("amino_acids.png")

st.image(image, use_column_width=True)

######################
# Input Text Box
######################

st.header('Enter amino acid sequence')
st.write("Put your sequence here 👇 (line break allowable | 1-letter code)")
sequence_input =  '' 
sequence_input = sequence_input.upper()

#sequence = st.sidebar.text_area("Sequence input", sequence_input, height=250)
sequence = st.text_area(sequence_input, height=250)
sequence = sequence.splitlines()
sequence = ''.join(sequence) # Concatenates list to string

## DNA nucleotide count
st.header('OUTPUT (DNA Nucleotide Count)')

### 1. Print dictionary
#st.subheader('1. Print dictionary')

def DNA_nucleotide_count(seq):
  d = dict([
            ('R',seq.count('R')),
            ('H',seq.count('H')),
            ('K',seq.count('K')),
            ('D',seq.count('D')),
            ('E',seq.count('E')),
            ('S',seq.count('S')),
            ('T',seq.count('T')),
            ('N',seq.count('N')),
            ('Q',seq.count('Q')),
            ('G',seq.count('G')),
            ('P',seq.count('P')),
            ('C',seq.count('C')),
            ('U',seq.count('U')),
            ('A',seq.count('A')),
            ('V',seq.count('V')),
            ('I',seq.count('I')),
            ('L',seq.count('L')),
            ('M',seq.count('M')),
            ('F',seq.count('F')),
            ('Y',seq.count('Y')),
            ('W',seq.count('W')),
            ])
  return d

X = DNA_nucleotide_count(sequence)

#X_label = list(X)
#X_values = list(X.values())



### 3. Display DataFrame
st.subheader('Display DataFrame')
df = pd.DataFrame.from_dict(X, orient='index')
df = df.rename({0: 'count'}, axis='columns')
df.reset_index(inplace=True)
df = df.rename(columns = {'index':'Amino acid'})
st.write(df)

### 4. Display Bar Chart using Altair
st.subheader('Display Bar chart')
p = alt.Chart(df).mark_bar().encode(
    x='Amino acid',
    y='count'
)
p = p.properties(
    width=alt.Step(30)  # controls width of bar.
)
st.write(p)

image2 = Image.open("1_letter_code.png")

st.image(image2, use_column_width=False)