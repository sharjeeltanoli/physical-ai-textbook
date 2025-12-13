import React, { useState, useEffect } from 'react';
import Content from '@theme-original/DocItem/Content';
import { useLocation } from '@docusaurus/router';
import UrduContent from '@site/src/components/UrduContent';

export default function ContentWrapper(props) {
  const [language, setLanguage] = useState('en');
  const [urduMarkdown, setUrduMarkdown] = useState(null);
  const location = useLocation();

  useEffect(() => {
    const saved = localStorage.getItem('siteLanguage') || 'en';
    setLanguage(saved);
    console.log('Initial language:', saved);
    if (saved === 'ur') loadUrdu();
  }, [location.pathname]);

  useEffect(() => {
    const handleChange = (e) => {
      console.log('Language changed event:', e.detail.language);
      setLanguage(e.detail.language);
      if (e.detail.language === 'ur') loadUrdu();
      else setUrduMarkdown(null);
    };
    window.addEventListener('languageChanged', handleChange);
    return () => window.removeEventListener('languageChanged', handleChange);
  }, [location.pathname]);

  const loadUrdu = async () => {
    console.log('Attempting to load Urdu content...');
    try {
      const path = location.pathname.replace(/^\/docs\//, '').replace(/\/$/, '');
      const response = await fetch(`/raw_docs/${path}.ur.md`);
      if (response.ok) {
        const content = await response.text();
        setUrduMarkdown(content);
        console.log('Urdu content loaded:', content.substring(0, 100) + '...'); // Log first 100 chars
      } else {
        console.error('Failed to fetch Urdu content, status:', response.status);
      }
    } catch (error) {
      console.error('Failed to load Urdu:', error);
    }
  };

  console.log('Render: language', language, 'urduMarkdown present:', !!urduMarkdown);
  if (language === 'ur' && urduMarkdown) {
    console.log('Rendering UrduContent with:', urduMarkdown.substring(0, 100) + '...');
    return <UrduContent content={urduMarkdown} />;
  }

  return <Content {...props} />;
}
