# Datasets
| Dataset | Link | Licença | Idioma | Criação | No experimento de geração |
| --- | --- | --- | --- | --- | --- |
| PorSimplesSent (NILC/USP) | https://github.com/sidleal/porsimplessent | CC BY 4.0 | pt-BR | humana (linguista), jornal e divulgação científica | sim: condições A, B, B2 (ids s01996, s01748, s02307, s02408, s02373, s00297, s02849) |
| TinyStories | https://huggingface.co/datasets/roneneldan/TinyStories | CDLA-Sharing-1.0 | EN | sintética (GPT-3.5/4), 3-4 anos | sim: C/D h01, h02, h03 |
| SimpleStories | https://huggingface.co/datasets/SimpleStories/SimpleStories | MIT | EN (+JA) | sintética (gpt-4o-mini), com metadados | sim: C/D h04, h05 |
| FairytaleQA | https://huggingface.co/datasets/WorkInTheDark/FairytaleQA | Apache-2.0 | EN | contos de Gutenberg + perguntas de educadores (K-8) | sim: C/D h06 |
| GlotStoryBook (subconjunto `por`) | https://huggingface.co/datasets/cis-lmu/GlotStoryBook | CC BY (parte NC) | pt-PT / pt-BR | humana (African Storybook, Storybooks Canada, LIDA, Pratham) | sim: E/E2 n01, n02, n03 |
| Storybooks Brasil (núcleo pt-BR do anterior) | https://global-asp.github.io/storybooks-brasil/ | CC BY 4.0 | pt-BR | humana (adaptação) | não (mesma fonte; ver GlotStoryBook) |
| historinhas | https://huggingface.co/datasets/Boakpe/historinhas | Apache-2.0 | pt-BR | sintética (modelo não documentado) | sim: E/E2 n04, n05 |
| TinyStories-Multilingual (pt) | https://huggingface.co/datasets/Dxniz/TinyStories-Multilingual | Apache-2.0 | pt (28 línguas) | sintética com juiz de segurança | sim: E/E2 n06 |
| FairytaleQA-translated-ptBR | https://huggingface.co/datasets/benjleite/FairytaleQA-translated-ptBR | Apache-2.0 | pt-BR | tradução DeepL do FairytaleQA | não: frases longas e literárias; complemento pedagógico |
| little-stories-en_US-pt_BR | https://huggingface.co/datasets/orion-research/little-stories-en_US-pt_BR | não declarada | pt-BR + EN | tradução automática do TinyStories, sem cartão | não: sem licença nem documentação |
| Children's Book Test (CBT) | https://huggingface.co/datasets/cam-cst/cbt | GFDL | EN | livros de Gutenberg (séc. XIX) | não: registro arcaico, muita fala |
| ROCStories / Story Cloze | https://huggingface.co/datasets/mintujupally/ROCStories (espelho); original https://cs.rochester.edu/nlp/rocstories/ (fora do ar em 23/10/2026) | indefinida | EN | crowdsourcing, adultos | não: conteúdo adulto e licença indefinida |
| VLibrasBD (pt-br2libras-gloss) | https://data.mendeley.com/datasets/ryj88ckjww/2 | CC BY-SA 4.0 | pt-BR + glosa Libras | glosa por dez intérpretes (LAVID/UFPB); frases de notícia, Wikidata e governo | não: mesmo registro de notícia do PorSimplesSent; útil como complemento (glosa ao lado da imagem) |

Artigos: TinyStories arXiv:2305.07759 · SimpleStories arXiv:2504.09184 · FairytaleQA https://aclanthology.org/2022.acl-long.34/ · CBT arXiv:1511.02301 · ROCStories arXiv:1604.01696 · VLibrasBD DOI 10.1016/j.dib.2025.111911 · PorSimplesSent https://aclanthology.org/C18-1034/
