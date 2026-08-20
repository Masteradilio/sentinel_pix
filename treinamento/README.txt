# EXP-010F-R2 + EXP-010G HQL rolling 180d

Arquivos gerados para substituir os HQLs atuais no HDFS:

/modelos_ml/nudan/nudan_hmo/

Parametros esperados nas caixinhas:
- DB_WORK=hmo_ml
- WINDOW_DAYS=180
- WINDOW_LAG_DAYS=1
- HQL_FILE=<nome_do_script.hql>

Ordem:
1. tb_pix_normais_qualified_raw_180d_v1
2. tb_pix_normais_qualified_sample_180d_v1
3. tb_pix_normais_qualified_sample_mbk_180d_v1
4. tb_pix_normais_dataset_ready_v1
5. tb_pix_dataset_v2_180d_v1
6. fork:
   - tb_pix_normais_qualified_audit_v1
   - tb_pix_normais_qualified_overlap_maf_audit_v1
