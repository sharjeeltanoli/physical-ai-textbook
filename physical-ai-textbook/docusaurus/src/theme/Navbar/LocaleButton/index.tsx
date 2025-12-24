
import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { useAlternatePageUtils } from '@docusaurus/theme-common/internal';
import { useLocation } from '@docusaurus/router';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

export default function LocaleButton() {
  const { i18n } = useDocusaurusContext();
  const { createUrl } = useAlternatePageUtils();
  const location = useLocation();

  const currentLocale = i18n.currentLocale;
  const otherLocale = i18n.locales.find((locale) => locale !== currentLocale);

  if (!otherLocale) {
    return null;
  }

  const alternatePageUrl = createUrl({
    locale: otherLocale,
    fullyQualified: false,
  });

  const localeLabel = otherLocale.toUpperCase();

  return (
    <Link to={alternatePageUrl} className={styles.localeButton}>
      {localeLabel}
    </Link>
  );
}
