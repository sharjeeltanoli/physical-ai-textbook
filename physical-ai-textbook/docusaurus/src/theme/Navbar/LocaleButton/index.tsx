
import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { useAlternatePageUtils } from '@docusaurus/theme-common/internal';
import { useLocation } from '@docusaurus/router';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

export default function LocaleButton() {
  const { i18n } = useDocusaurusContext();
  const { alternatePageLocalizations } = useAlternatePageUtils();
  const location = useLocation();

  const currentLocale = i18n.currentLocale;
  const otherLocale = i18n.locales.find((locale) => locale !== currentLocale);

  if (!otherLocale || !alternatePageLocalizations) {
    return null;
  }

  // Find the alternate page for the other locale
  const alternatePage = alternatePageLocalizations.find(
    (altPage) => altPage.locale === otherLocale
  );

  // If no alternate page is found for the other locale, don't render the button
  if (!alternatePage) {
    return null;
  }

  const localeLabel = otherLocale.toUpperCase();

  return (
    <Link to={alternatePage.path} className={styles.localeButton}>
      {localeLabel}
    </Link>
  );
}
