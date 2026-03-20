Me ajude com um tema complexo sobre Ciência de Dados.

Sou um cientista de dados júnior, que trabalho em um banco público comercial no Brasil. Preciso elaborar um modelo de ML que sirva para detectar anomalias em transações de PIX feitas pelos clientes do banco com a finalidade de detectar as transações mais fora do normais e sinalizar elas, antes que uma fraude aconteça, dando a oportunidade de um analista humano analisar tudo e aprovar ou reprovar a transação.

Esse modelo pode ser, na verdade, um ensemble de modelos (desde que isso seja mais útil do que um modelo único) e junto desse modelo quero impor também um conjunto de regras que advém do comportamento já conhecido de fraudadores, se identificarmos algumas dessas características na transação, o score final de classificação deve receber pesos. As regras seriam essas aqui:

"""
AGRAVANTES:
    1. PIX em < 30min: 1 tx=peso 1, 2+ tx=peso 2
    2. Razão PIX/Limite: 0.4-0.59=1, 0.6-0.79=2, 0.8+=3
    3. Idade: 60-65=1, 66-75=2, 76+=3
    4. Tempo relacionamento: 61-90 dias=1, 31-60 dias=2, 0-30 dias=3
    5. Conta laranja: peso 3
    6. Chave aleatória: peso 2
    7. Horário noturno: peso 3
    8. Velocity checks: peso 2-4
    9. Topaz: peso 2-5

ATENUANTES:
    - Autorização prévia: reduz 50% do score final
"""

Em anexo você tem 3 bases de dados, das quais você não precisa ler tudo, carregue em sua memória somente 100 linhas de cada coluna de cada arquivo, só para entender a estrutura dos dados reais que tenho disponível para trabalhar esse modelo. 

O primeiro CSV, é um arquivo com dados de transações normais de PIX dos clientes do banco. O Segundo é um CSV com casos já conhecidos de fraude consumadas no banco, eles possuem a label "is_fraud" ou algo do tipo, e é binário com 0 para não e 1 para sim. O terceiro é um conjunto de dados de utilização do aplicativo mobile dos clientes, de onde também se pode extrair o score Topaz, que é um dos agravantes (vai de zero a 5, sendo 5 o pior) e é um score de ML que traz o risco do aparelho do cliente ser fruto de hackeamento ou coisa do tipo.

Além do modelo ou modelos de ML para detectar transações que possuem estatísticas fora do padrão ou são muito semelhantes às transações fraudulentas, preciso detectar componentes de comportamento estranho dos clientes (behavioral analytics) e de engenharia social pois há fraudes que os próprios clientes estão enviando o dinheiro, mas na verdade eles estão sendo induzidos ou enganados por pessoas que se passam por falsos funcionários ou mentiras semelhantes.

Quais ideias você pode me fornecer para criar esse modelo de detecção em tempo real de transações PIX fraudulentas?