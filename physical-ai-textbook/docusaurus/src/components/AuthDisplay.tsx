import React, {useEffect, useState} from 'react';
import Link from '@docusaurus/Link';
import {useHistory} from '@docusaurus/router';
import BetterAuth from '../utils/betterAuth';

function AuthDisplay({ className }: { className?: string }) {
  const [currentUser, setCurrentUser] = useState(null);
  const history = useHistory();

  useEffect(() => {
    setCurrentUser(BetterAuth.getCurrentUser());
  }, []);

  const handleSignOut = () => {
    BetterAuth.signOut();
    setCurrentUser(null);
    history.push('/');
  };

  const linkClassName = className || "navbar__item navbar__link";

  if (currentUser) {
    return (
      <Link
        className={linkClassName}
        to="#" // Assuming a profile page will be created
        onClick={handleSignOut}>
        Sign Out ({currentUser.username})
      </Link>
    );
  }

  return (
    <Link
      className={linkClassName}
      to="/auth">
      Sign In
    </Link>
  );
}

export default AuthDisplay;
