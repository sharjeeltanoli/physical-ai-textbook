import type {ReactNode} from 'react';
import React, {useState, useEffect} from 'react';
import Layout from '@theme/Layout';
import {useHistory} from '@docusaurus/router'; // Import useHistory for redirection
import BetterAuth from '../utils/betterAuth'; // Import BetterAuth from utility file


function AuthPage(): ReactNode {
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false); // New state for password visibility
  const [gender, setGender] = useState('');
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
      const backgroundInfo = {gender, softwareExperience, hardwareExperience};
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
      <div style={{display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 'calc(100vh - 60px)'}}>
        <div style={{width: '100%', maxWidth: '400px', padding: '2rem', border: '1px solid #ccc', borderRadius: '8px', margin: '1rem'}}>
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
              <div style={{position: 'relative'}}> {/* This div now acts as the positioning context for the button */}
                <input
                  type={showPassword ? 'text' : 'password'} // Dynamic type
                  id="password"
                  name="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  style={{width: '100%', padding: '0.5rem', paddingRight: '40px'}} /* Added paddingRight */
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((prev) => !prev)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  style={{
                    position: 'absolute',
                    right: '10px',
                    top: '50%', // Centered vertically relative to its parent (the new div)
                    transform: 'translateY(-50%)', // Adjust for button's own height
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    padding: '0',
                    color: '#333', // Dark grey color, consistent with default text
                    height: '20px', // Explicitly set height for better centering calculation
                    display: 'flex', // Use flex to center SVG if it had issues
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                >
                  {showPassword ? (
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.05 18.05 0 0 1 4.58-5.89M2 2l20 20"></path>
                      <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.08 2.92"></path>
                      <circle cx="12" cy="12" r="3"></circle>
                    </svg>
                  ) : (
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                      <circle cx="12" cy="12" r="3"></circle>
                    </svg>
                  )}
                </button>
              </div>
            </div>
            {isSignUp && (
              <>
                <div style={{marginBottom: '1rem'}}>
                  <label htmlFor="gender">Gender</label>
                  <select
                    id="gender"
                    name="gender"
                    value={gender}
                    onChange={(e) => setGender(e.target.value)}
                    required
                    style={{width: '100%', padding: '0.5rem'}}
                  >
                    <option value="" disabled>Select your gender</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="prefer_not_to_say">Prefer not to say</option>
                  </select>
                </div>
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