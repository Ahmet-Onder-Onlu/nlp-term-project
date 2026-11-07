# Yol Haritası (Roadmap)

Hedef: RETECO Track 1 (TEMPO) 1a + 1b için resmi BM25'i (dev 0.115 / 0.118) belirgin şekilde
geçen, ablation'larla desteklenmiş bir sistem ve ACL formatında makale.

| Faz | İçerik | Çıktı | Durum |
|---|---|---|---|
| 0 | Görev analizi, veri şeması, proje iskeleti, makale şablonu | bu repo | ✅ |
| 1 | Tüm Track 1 verisini indir, E0 BM25'i resmi baseline ile karşılaştır (±0.01) | `results/metrics/e0_*` | ⏳ |
| 2 | Metadata analizi: doküman yıl kapsamı, guidance istatistikleri (intent dağılımı) | makale §3.1 tablo | |
| 3 | E1 temporal prior, λ'yı train'de ayarla | E1 sonuçları | |
| 4 | Dense indeks (GPU: Colab/Kaggle) + Qdrant + KG → E2 | E2 sonuçları | |
| 5 | LLM sorgu analizcisi (guidance-distilled), analizci doğruluğu dev'de ölçülür | E3a | |
| 6 | Cross-encoder + LLM listwise rerank → E3 | E3 sonuçları | |
| 7 | (Opsiyonel) RAPTOR tarzı domain-içi küme özetleri kanalı; reasoning embedder (ör. ReasonIR) | E4 | |
| 8 | Ablation, domain ve temporal-intent analizi, hata analizi, figürler | makale §5 | |
| 9 | Makale yazımı, Limitations, kod yayını; Ocak 2027 test penceresinde submission | PDF + runs | |

## Kurallar
- Ayarlama **train**'de, raporlama **dev**'de. Guidance/gold hiçbir zaman değerlendirilen sorgu için girdi değildir.
- Her deney bir YAML config'tir; sonuç JSON'ları `results/metrics/` altına düşer, makale tabloları buradan üretilir.
- LLM çağrıları `.cache/llm/` altında önbelleklenir (tekrar üretilebilirlik + maliyet).
