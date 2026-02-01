# TEFAS Sim Dashboard

## Sprint 4 Özellikleri

- Arayüz tamamen Türkçe hale getirildi.
- Fon filtresi (AUM, gözlem günü, eksik gün toleransı, fon adı/kodu arama) eklendi.
- Fon özeti tablosu zenginleştirildi (getiri, vol, Sharpe, MDD, en iyi/en kötü gün).
- Stratejilere Risk Parity ve Min Variance eklendi.
- Rolling metrik grafikleri (volatilite, Sharpe, MDD) eklendi.
- Korelasyon ısı haritası ve en iyi/kötü çiftler listesi eklendi.
- Fon karşılaştırma bölümü (equity eğrisi, histogram, KPI) eklendi.
- İndirme seçenekleri genişletildi.

## Çalıştırma

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Testler

```bash
pytest -q
```
