# Política de Engenharia — Acme Tecnologia

> Documento fictício, criado apenas como base de testes deste projeto.

## Code review

Todo pull request precisa de aprovação de pelo menos **dois revisores**, sendo
um deles obrigatoriamente membro do time responsável pelo módulo alterado.
Pull requests com mais de 400 linhas alteradas devem ser quebrados antes da
revisão, salvo em migrações automáticas geradas por ferramenta.

O tempo máximo acordado para a primeira resposta em um pull request é de
**24 horas úteis**. Passado esse prazo, o autor pode acionar o canal
`#eng-review-sos`.

## Cobertura de testes

O portão de qualidade exige cobertura mínima de **80% de linhas** e **70% de
ramos** nos módulos de domínio. Módulos de infraestrutura (adapters, clientes
HTTP, configuração) estão isentos do portão, mas precisam de ao menos um teste
de integração.

Testes de mutação com PIT rodam semanalmente na branch `main`. O índice de
mutantes mortos não pode cair abaixo de **65%** por duas semanas seguidas.

## Deploy

O deploy em produção acontece por *trunk based development*, com feature flags
para funcionalidades incompletas. Existe janela de congelamento (*code freeze*)
das **18h de sexta-feira até as 8h de segunda-feira**, exceto para correções
classificadas como severidade 1.

Toda release precisa de um plano de rollback documentado no próprio pull
request. Rollback automático é acionado quando a taxa de erro 5xx ultrapassa
**2% por 5 minutos consecutivos**.

## Convenções de commit

O padrão adotado é Conventional Commits, com os tipos `feat`, `fix`, `docs`,
`refactor`, `test` e `chore`. O versionamento segue SemVer e é gerado
automaticamente a partir dos commits na `main`.
