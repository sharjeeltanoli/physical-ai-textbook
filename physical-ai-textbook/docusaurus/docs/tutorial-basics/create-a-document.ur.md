---
sidebar_position: 2
---

# ایک دستاویز بنائیں

دستاویزات **صفحات کا ایک گروپ** ہیں جو آپس میں منسلک ہوتے ہیں بذریعہ:

- ایک **سائیڈبار**
- **پچھلی/اگلی نیویگیشن**
- **ورژننگ**

## اپنی پہلی دستاویز بنائیں

`docs/hello.md` پر ایک مارک ڈاؤن فائل بنائیں:

md title="docs/hello.md"
# Hello

This is my **first Docusaurus document**!

ایک نئی دستاویز اب [http://localhost:3000/docs/hello](http://localhost:3000/docs/hello) پر دستیاب ہے۔

## سائیڈبار کو ترتیب دیں

Docusaurus خود بخود `docs` فولڈر سے **ایک سائیڈبار بناتا ہے**۔

سائیڈبار کے لیبل اور پوزیشن کو حسب ضرورت بنانے کے لیے میٹا ڈیٹا شامل کریں:

```md title="docs/hello.md" {1-4}
---
sidebar_label: 'Hi!'
sidebar_position: 3
---

# Hello

This is my **first Docusaurus document**!
```

اپنی سائیڈبار کو `sidebars.js` میں صراحتاً (explicitly) بنانا بھی ممکن ہے:

```js title="sidebars.js"
export default {
  tutorialSidebar: [
    'intro',
    // highlight-next-line
    'hello',
    {
      type: 'category',
      label: 'Tutorial',
      items: ['tutorial-basics/create-a-document'],
    },
  ],
};
```
