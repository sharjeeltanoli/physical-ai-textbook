import React, { useState, useEffect } from 'react';

export default function TranslateButton({ className }) {
  const [language, setLanguage] = useState('en');

  useEffect(() => {
    let saved = localStorage.getItem('siteLanguage');
    if (!saved || saved === 'ur') { // Force 'en' if not set or if it's 'ur'
      saved = 'en';
      localStorage.setItem('siteLanguage', 'en');
    }
    setLanguage(saved);
  }, []);

  const toggleLanguage = () => {
    const newLang = language === 'en' ? 'ur' : 'en';
    setLanguage(newLang);
    localStorage.setItem('siteLanguage', newLang);
    window.dispatchEvent(new CustomEvent('languageChanged', { detail: { language: newLang } }));
  };

  return (
    <button onClick={toggleLanguage} className={className}>
      {language === 'en' ? 'اردو' : 'English'}
    </button>
  );
}