# Laboratório Técnico 05: Treinamento Fim-a-Fim do Transformer

**Aluno:** Andreas Carvalho  
**Disciplina:** Tópicos em Inteligência Artificial  
**Professor:** Dimmy Magalhães  

---

## Descrição

Este projeto dá continuidade ao Laboratório 04, utilizando a arquitetura do **Transformer Encoder-Decoder implementada "from scratch"**, agora conectada a um **dataset real do Hugging Face**, com o objetivo de realizar o **treinamento fim-a-fim (forward, loss, backward e step)**.

O foco deste laboratório não é obter traduções perfeitas, mas sim demonstrar que a arquitetura construída é capaz de **aprender**, evidenciado pela **queda significativa da função de perda (Loss)** ao longo das épocas.

---

## Objetivos do projeto

- Integrar um dataset real do Hugging Face ao pipeline do modelo
- Utilizar tokenização automática com a biblioteca `transformers`
- Converter texto em representações numéricas (IDs)
- Aplicar padding e tokens especiais `<START>` e `<EOS>`
- Implementar o loop completo de treinamento com:
  - Forward pass
  - Cálculo da Loss (CrossEntropy)
  - Backpropagation
  - Atualização de pesos (Adam)
- Monitorar a convergência da Loss ao longo das épocas
- Validar o funcionamento do modelo com um **teste de overfitting**

---

## Estrutura implementada

### Dataset

Foi utilizado o dataset:

- https://huggingface.co/datasets/bentrevett/multi30k

Apenas um subconjunto de **1.000 pares de frases** foi utilizado para garantir execução viável em CPU, conforme orientação do laboratório.

---

### Tokenização

- Tokenizador utilizado: `bert-base-multilingual-cased`
- Conversão de frases para listas de IDs
- Adição de:
  - `<START>` no início da sequência de entrada do Decoder
  - `<EOS>` no final da sequência alvo
- Aplicação de padding para uniformizar o tamanho dos batches

---

### Modelo (Reutilizado do Lab 04)

A arquitetura do Transformer foi mantida, contendo:

- **Scaled Dot-Product Attention**
- **Multi-Head Attention**
- **Masked Self-Attention**
- **Cross-Attention**
- **Add & Norm**
- **Feed-Forward Network (FFN)**
- **Positional Encoding**
- **Encoder e Decoder com múltiplas camadas**

---

### Treinamento

O treinamento segue o fluxo clássico:

1. Entrada passa pelo Encoder  
2. Saída do Encoder é utilizada pelo Decoder  
3. Decoder recebe sequência deslocada (teacher forcing)  
4. Cálculo da Loss com `CrossEntropyLoss`  
5. Aplicação de `loss.backward()`  
6. Atualização dos pesos com `Adam`  

A Loss é impressa a cada época, sendo possível observar sua redução ao longo do treinamento.

---

### Overfitting Test (Prova de Fogo)

Após o treinamento:

- Uma frase do conjunto de treino é selecionada
- O modelo gera a tradução de forma **auto-regressiva**
- O resultado é comparado com a tradução real

O modelo foi capaz de reproduzir a tradução corretamente, indicando que:

- os gradientes estão fluindo corretamente  
- a arquitetura está funcionando  
- o treinamento foi efetivo  

---

## Tecnologias utilizadas

- Python
- PyTorch
- Hugging Face (`datasets` e `transformers`)

---

## Arquivos do projeto

- `hugging.py` — implementação completa do Transformer + treinamento
- `requirements.txt` — dependências
- `README.md` — documentação

---

## Execução

Instalar dependências:


pip install -r requirements.txt


Executar o projeto:


python hugging.py


---

## Tempo de execução

O tempo total de execução é exibido ao final do programa.

O treinamento foi realizado em CPU, com tempo aproximado de alguns minutos, dependendo da máquina.

---

## Referências teóricas

- Vaswani, A. et al. *Attention Is All You Need*. NeurIPS, 2017.

Os conceitos foram baseados nas aulas da disciplina e nos laboratórios anteriores, especialmente na implementação do Transformer no Lab 04.

---

## Uso de Inteligência Artificial

Durante o desenvolvimento deste projeto, utilizei ferramentas de IA, especialmente o ChatGPT, como apoio complementar, conforme permitido pelo enunciado do laboratório :contentReference[oaicite:1]{index=1}.

O uso ocorreu principalmente em:

- auxílio na integração com o dataset do Hugging Face
- apoio na utilização de tokenizadores (`AutoTokenizer`)
- esclarecimento de dúvidas conceituais sobre o fluxo de treinamento
- ajuda na implementação do pipeline de dados (tokenização, padding, batches)
- suporte na organização do loop de treinamento (forward, loss, backward, step)
- auxílio em funções específicas do PyTorch
- revisão e ajuste de partes da lógica do código

O uso de IA foi significativo no suporte à lógica e estrutura do código, porém:

- a arquitetura do Transformer foi mantida conforme implementada no Lab 04
- o funcionamento do modelo foi compreendido e validado manualmente
- todas as decisões finais foram revisadas antes da entrega

A IA foi utilizada como ferramenta de apoio, e não como substituição do aprendizado.

---