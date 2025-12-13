---
sidebar_position: 5
---

# اپنی سائٹ کو Deploy کریں

Docusaurus ایک **static-site-generator** ہے (جسے **[Jamstack](https://jamstack.org/)** بھی کہا جاتا ہے)۔

یہ آپ کی سائٹ کو سادہ **static HTML، JavaScript اور CSS فائلوں** کی صورت میں بناتا ہے۔

## اپنی سائٹ Build کریں

اپنی سائٹ کو **پروڈکشن کے لیے** Build کریں:

bash
npm run build

static فائلیں `build` فولڈر میں generate ہوتی ہیں۔

## اپنی سائٹ کو Deploy کریں

اپنی پروڈکشن build کو مقامی طور پر ٹیسٹ کریں:

```bash
npm run serve
```

`build` فولڈر اب [http://localhost:3000/](http://localhost:3000/) پر serve کیا جاتا ہے۔

اب آپ `build` فولڈر کو **تقریباً کہیں بھی** آسانی سے، **مفت میں** یا بہت کم لاگت پر deploy کر سکتے ہیں (**[Deployment Guide](https://docusaurus.io/docs/deployment)** پڑھیں)۔
