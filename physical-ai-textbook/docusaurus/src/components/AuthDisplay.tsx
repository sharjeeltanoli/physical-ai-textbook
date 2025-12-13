import React, {useEffect, useState} from 'react';
import Link from '@docusaurus/Link';
import {useHistory} from '@docusaurus/router';
import BetterAuth from '../utils/betterAuth';

function AuthDisplay() {
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

  if (currentUser) {
    return (
      <Link
        className="navbar__item navbar__link"
        to="#" // Assuming a profile page will be created
        onClick={handleSignOut}>
        Sign Out ({currentUser.username})
      </Link>
    );
  }

  return (
    <Link
      className="navbar__item navbar__link"
      to="/auth">
      Sign In
    </Link>
  );
}

export default AuthDisplay;
