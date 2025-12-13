---
sidebar_position: 1
---

# ڈاکس ورژنز کا انتظام کریں

Docusaurus آپ کے ڈاکس کے متعدد ورژنز کا انتظام کر سکتا ہے۔

## ایک ڈاکس ورژن بنائیں

اپنے پروجیکٹ کا ورژن 1.0 جاری کریں:

bash
npm run docusaurus docs:version 1.0

`docs` فولڈر کو `versioned_docs/version-1.0` میں کاپی کیا جاتا ہے اور `versions.json` تخلیق کی جاتی ہے۔

آپ کے ڈاکس کے اب 2 ورژنز ہیں:

- `1.0` `http://localhost:3000/docs/` پر ورژن 1.0 کے ڈاکس کے لیے
- `current` `http://localhost:3000/docs/next/` پر **آنے والے، غیر جاری کردہ ڈاکس** کے لیے

## ایک ورژن ڈراپ ڈاؤن شامل کریں

ورژنز کے درمیان بغیر کسی رکاوٹ کے نیویگیٹ کرنے کے لیے، ایک ورژن ڈراپ ڈاؤن شامل کریں۔

`docusaurus.config.js` فائل میں ترمیم کریں:

```js title="docusaurus.config.js"
export default {
  themeConfig: {
    navbar: {
      items: [
        // highlight-start
        {
          type: 'docsVersionDropdown',
        },
        // highlight-end
      ],
    },
  },
};
```

ڈاکس ورژن ڈراپ ڈاؤن آپ کے نیو بار میں ظاہر ہوتا ہے:

![Docs Version Dropdown](./img/docsVersionDropdown.png)

## موجودہ ورژن کو اپ ڈیٹ کریں

ورژنڈ ڈاکس کو ان کے متعلقہ فولڈر میں ترمیم کرنا ممکن ہے:

- `versioned_docs/version-1.0/hello.md` `http://localhost:3000/docs/hello` کو اپ ڈیٹ کرتا ہے
- `docs/hello.md` `http://localhost:3000/docs/next/hello` کو اپ ڈیٹ کرتا ہے
