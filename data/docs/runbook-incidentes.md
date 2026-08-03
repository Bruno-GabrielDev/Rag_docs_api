# Runbook de Incidentes — Acme Tecnologia

> Documento fictício, criado apenas como base de testes deste projeto.

## Classificação de severidade

| Severidade | Critério | Tempo de resposta |
| --- | --- | --- |
| SEV1 | Indisponibilidade total ou perda de dados de clientes | 15 minutos |
| SEV2 | Funcionalidade crítica degradada, sem alternativa | 1 hora |
| SEV3 | Funcionalidade não crítica degradada ou com contorno | 8 horas úteis |
| SEV4 | Problema cosmético ou de baixo impacto | próximo ciclo |

## Fluxo de atendimento

1. Quem identifica o incidente abre o canal `#inc-<data>-<slug>` e assume o
   papel de **comandante do incidente** até que alguém do time de plantão
   assuma formalmente.
2. O comandante não executa correções: coordena, registra a linha do tempo e
   comunica os interessados. Essa separação existe para que a comunicação não
   pare quando quem está corrigindo mergulha no problema.
3. Comunicação externa é responsabilidade exclusiva do time de Suporte, a
   partir do texto aprovado pelo comandante.

## Escalonamento

Incidentes SEV1 são escalonados para a liderança de engenharia em até
**30 minutos** sem sinal de resolução. Incidentes SEV2 escalonam em 2 horas.
O plantão funciona em turnos de 7 dias, com revezamento às quartas-feiras.

## Post-mortem

Todo SEV1 e SEV2 exige post-mortem escrito em até **5 dias úteis**. O
post-mortem é sem culpados (*blameless*): descreve causas sistêmicas, nunca
nomes de pessoas responsabilizadas. Cada post-mortem precisa gerar no mínimo
uma ação corretiva com dono e prazo definidos.
