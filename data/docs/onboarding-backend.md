# Onboarding — Time de Backend

> Documento fictício, criado apenas como base de testes deste projeto.

## Stack

Os serviços de backend usam Python 3.11 com FastAPI para APIs síncronas e
Celery com Redis para processamento assíncrono. O banco transacional é
PostgreSQL 16, e o armazenamento de objetos é S3. Serviços legados em Java 17
com Spring Boot ainda existem no domínio de faturamento e estão em migração.

## Ambiente local

O ambiente sobe com `docker compose up -d`, que provisiona PostgreSQL, Redis e
o LocalStack. As migrações são aplicadas com Alembic através de
`make migrate`. As credenciais de desenvolvimento ficam no `.env.example` e
nunca devem ser substituídas por credenciais reais.

O acesso ao banco de homologação é liberado apenas via VPN, e somente em modo
leitura para pessoas fora do time de plataforma.

## Primeira semana

O primeiro pull request esperado de uma pessoa recém-chegada é a correção de
uma issue marcada com a etiqueta `good first issue`. A meta é abrir esse pull
request até o **quinto dia útil**, com apoio de um padrinho designado no
primeiro dia.

Reuniões fixas do time: daily às 9h30 (15 minutos), refinamento às terças às
14h e retrospectiva quinzenal às sextas às 16h.

## Padrões de projeto

Os serviços seguem arquitetura hexagonal: o domínio não importa nada de
framework, e adaptadores ficam isolados na borda. Injeção de dependência é
feita por construtor, sem contêiner mágico. Toda chamada a serviço externo
precisa de timeout explícito e política de retry com backoff exponencial.
