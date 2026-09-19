"""dataset."""

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def carregar_uci():
    """Baixa e lê o arquivo oficial Bank Marketing sem criar cópias locais:"""
    with urlopen(UCI_URL, timeout=30) as resposta:
        conteudo = resposta.read()
    with ZipFile(BytesIO(conteudo)) as pacote:
        with pacote.open("bank.zip") as zip_interno:
            with ZipFile(BytesIO(zip_interno.read())) as arquivo:
                caminho = next(nome for nome in arquivo.namelist() if nome.endswith("bank-full.csv"))
                with arquivo.open(caminho) as csv:
                    return pd.read_csv(csv, sep=";")
