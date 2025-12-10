import type {ReactNode} from 'react';
import React, {useState, useEffect} from 'react';
import Layout from '@theme/Layout';
import {useHistory} from '@docusaurus/router'; // Import useHistory for redirection
import BetterAuth from '../utils/betterAuth'; // Import BetterAuth from utility file


function AuthPage(): ReactNode {
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [softwareExperience, setSoftwareExperience] = useState('beginner');
  const [hardwareExperience, setHardwareExperience] = useState('beginner');
  const [loading, setLoading] = useState(false);
  const history = useHistory();

  useEffect(() => {
    // Check if user is already logged in
    if (BetterAuth.getCurrentUser()) {
      history.push('/'); // Redirect to home if logged in
    }
  }, [history]);

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    let result;
    if (isSignUp) {
      const backgroundInfo = {softwareExperience, hardwareExperience};
      result = await BetterAuth.signUp(email, password, backgroundInfo);
    } else {
      result = await BetterAuth.signIn(email, password);
    }

    if (result.success) {
      alert(`${isSignUp ? 'Sign up' : 'Sign in'} successful!`);
      history.push('/'); // Redirect to home on success
    } else {
      alert(`Authentication failed: ${result.message}`);
    }
    setLoading(false);
  };

  return (
    <Layout title="Sign In / Sign Up">
      <div style={{display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh'}}>
        <div style={{width: '400px', padding: '2rem', border: '1px solid #ccc', borderRadius: '8px'}}>
          <h2>{isSignUp ? 'Sign Up' : 'Sign In'}</h2>
          <form onSubmit={handleAuth}>
            <div style={{marginBottom: '1rem'}}>
              <label htmlFor="email">Email</label>
              <input
                type="email"
                id="email"
                name="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                style={{width: '100%', padding: '0.5rem'}}
              />
            </div>
            <div style={{marginBottom: '1rem'}}>
              <label htmlFor="password">Password</label>
              <input
                type="password"
                id="password"
                name="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                style={{width: '100%', padding: '0.5rem'}}
              />
            </div>
            {isSignUp && (
              <>
                <div style={{marginBottom: '1rem'}}>
                  <label htmlFor="softwareExperience">Software Experience</label>
                  <select
                    id="softwareExperience"
                    name="softwareExperience"
                    value={softwareExperience}
                    onChange={(e) => setSoftwareExperience(e.target.value)}
                    style={{width: '100%', padding: '0.5rem'}}
                  >
                    <option value="beginner">Beginner</option>
                    <option value="intermediate">Intermediate</option>
                    <option value="advanced">Advanced</option>
                  </select>
                </div>
                <div style={{marginBottom: '1rem'}}>
                  <label htmlFor="hardwareExperience">Hardware Experience</label>
                  <select
                    id="hardwareExperience"
                    name="hardwareExperience"
                    value={hardwareExperience}
                    onChange={(e) => setHardwareExperience(e.target.value)}
                    style={{width: '100%', padding: '0.5rem'}}
                  >
                    <option value="beginner">Beginner</option>
                    <option value="intermediate">Intermediate</option>
                    <option value="advanced">Advanced</option>
                  </select>
                </div>
              </>
            )}
            <button
              type="submit"
              disabled={loading}
              style={{width: '100%', padding: '0.5rem', backgroundColor: '#25c2a0', color: 'white', border: 'none', borderRadius: '4px', cursor: loading ? 'not-allowed' : 'pointer'}}
            >
              {loading ? 'Loading...' : (isSignUp ? 'Sign Up' : 'Sign In')}
            </button>
          </form>
          <div style={{marginTop: '1rem', textAlign: 'center'}}>
            <button onClick={() => setIsSignUp(!isSignUp)} style={{background: 'none', border: 'none', color: '#25c2a0', cursor: 'pointer'}}>
              {isSignUp ? 'Already have an account? Sign In' : "Don't have an account? Sign Up"}
            </button>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default AuthPage;