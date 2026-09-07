# fable5.1 关键引文 DOI/题名核验

核验采用 Crossref REST、OpenAlex REST 与 arXiv API（2026-09-05）。这里的“未检出”表示在上述公开元数据检索中没有得到与作者、年份、题名和期刊同时匹配的记录，不等同于证明不存在；最终稿不得把未检出条目写成已证实 prior art。

| fable5.1 条目 | 核验结果 | 证据/修订 |
|---|---|---|
| Ye, Do, Zeng & Lambert, *Water Research* (2022), PINN hydraulic transient | **真实，可保留** | Crossref: *Physics-informed neural networks for hydraulic transient analysis in pipeline systems*; authors Jiawei Ye, Nhu Cuong Do, Wei Zeng, Martin Lambert; DOI `10.1016/j.watres.2022.118828`; published 2022-08; Water Research. 该文是管道水力瞬变 PINN prior art，不应扩大为多簇裂缝系统结果。 |
| Molinaro, Yang, Engquist & Mishra, “NIO” (ICML 2023) | **真实预印本；会议/DOI需单独核实** | arXiv API: *Neural Inverse Operators for Solving PDE Inverse Problems*, arXiv:2301.11167v2 (2023-06-03), authors Roberto Molinaro, Yunan Yang, Björn Engquist, Siddhartha Mishra。摘要明确 NIO 是 DeepONet 与 FNO 组合、用于 PDE inverse maps。引用时写 arXiv 版本，除非取得正式 ICML 论文元数据；不要写成针对井口单点水击。 |
| Bartolucci et al., NeurIPS 2023 | **真实，可保留但需收窄论断** | Crossref/OpenAlex: *Representation Equivalent Neural Operators: a Framework for Alias-free Operator Learning*, authors Francesca Bartolucci, Emmanuel De Bézenac, Bogdan Raonic, Roberto Molinaro, Siddhartha Mishra, Rima Alaifari; NeurIPS 36 (2023); DOI `10.52202/075280-3051`。论文主题是 alias-free/representation-equivalent operator；fable 中“证明标准 FNO 对非带限输入的全部误差结论”应改为“讨论/缓解离散表示与混叠问题”，避免过度归因。 |
| Liu, Xu, Cao & Zhang, *JCP* (2024), “HANO” | **未核验，不应按现状引用** | Crossref `query.bibliographic`、OpenAlex 搜索未得到题名、作者和 JCP DOI 的同时匹配；普通“HANO”结果主要指非学术缩写。需作者提供准确题名/DOI后再纳入。 |
| Zhu et al., *CMAME* (2023), neural-operator OOD/Wasserstein claim | **未核验** | Crossref 题名检索返回无关的 “Neural operator search” (Pattern Recognition DOI `10.1016/j.patcog.2022.109215`) 等，未发现与 fable 叙述相符的 Zhu–CMAME 2023 条目。不得保留“证明外推误差随 2-Wasserstein 距离超线性”这一具体归因。 |
| Benitez et al., *JCP* (2024), neural-operator OOD theorem | **未核验** | 未检出与作者、期刊、年份和所述 Wasserstein 外推结论同时匹配的 Crossref/OpenAlex 记录；应删除或改为无作者归因的综述性假设，直至补足 DOI。 |
| Lippe et al., NeurIPS 2023, long-horizon neural-operator energy drift | **未核验** | 未检出与 fable 所述“自回归长时程能量漂移”相符的 Lippe–NeurIPS 2023 论文及 DOI；不可据此支撑具体失效机理。可改引有明确出处的 PDE surrogate 长时滚动误差文献。 |
| Kaltenbach, Perdikaris & Koutsourelakis, *Computational Mechanics* (2023), invertible Bayesian operator | **未核验** | Crossref author/title 检索未返回匹配记录；“可逆贝叶斯算子”表述和作者组合缺乏可核验 DOI。暂列待核，不作为已发表 prior art。 |
| Liu 等 2024 HANO、Zhu 2023 CMAME、Benitez 2024 JCP、Lippe 2023 NeurIPS、Kaltenbach 2023 CM | **统一处理** | 在最终报告 benchmark 表中标记“citation unverified / remove until DOI supplied”，避免“真实公开成果”声明与证据冲突。 |

## 可安全引用的替代基础文献

- Raissi, Perdikaris & Karniadakis, *J. Comput. Phys.* 378 (2019), 686–707, DOI `10.1016/j.jcp.2018.10.045`（PINN 基础）。
- Lu et al., *Nature Machine Intelligence* 3 (2021), 218–229, DOI `10.1038/s42256-021-00302-5`（DeepONet）。
- Li et al., ICLR 2021, “Fourier Neural Operator for Parametric Partial Differential Equations”（正式会议条目，引用前按 OpenReview/DBLP 核对版本）。
- Kovachki et al., *JMLR* 24 (2023), “Neural Operator: Learning Maps Between Function Spaces”（统一神经算子综述/理论框架）。

