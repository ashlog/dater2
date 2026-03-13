import { useEffect, useState, useRef, useCallback } from 'react';
import { QRCodeSVG } from 'qrcode.react';

interface Profile {
  code: string;
  username: string;
  score: number;
  link: string;
  full_name?: string;
  imageDataUri?: string;
  cdnUrl?: string;
  name?: string;
  bio?: string;
  caption?: string;
  opener?: string;
  openerLoading?: boolean;
  imageLoading?: boolean;
}

export function App() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/profiles')
      .then(r => r.json())
      .then((data: Profile[]) => { setProfiles(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const updateProfile = useCallback((code: string, update: Partial<Profile>) => {
    setProfiles(prev => prev.map(p => p.code === code ? { ...p, ...update } : p));
  }, []);

  if (loading) return <div className="loading"><span className="spinner" /> Loading profiles...</div>;

  return (
    <div className="app">
      <header>
        <h1>Instagram <span>Openers</span></h1>
        <p className="subtitle">{profiles.length} profiles ranked by SigLIP image classifier</p>
      </header>
      <div className="grid">
        {profiles.map((p, i) => (
          <ProfileCard key={p.code} profile={p} rank={i + 1} updateProfile={updateProfile} />
        ))}
      </div>
    </div>
  );
}

function ProfileCard({ profile, rank, updateProfile }: {
  profile: Profile;
  rank: number;
  updateProfile: (code: string, update: Partial<Profile>) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const loadedRef = useRef(false);

  useEffect(() => {
    if (loadedRef.current || profile.imageDataUri) return;
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !loadedRef.current) {
        loadedRef.current = true;
        obs.disconnect();
        updateProfile(profile.code, { imageLoading: true });
        Promise.all([
          fetch(`/api/image/${profile.code}`).then(r => r.ok ? r.json() : null),
          fetch(`/api/profile/${profile.username}`).then(r => r.ok ? r.json() : null),
          fetch(`/api/caption/${profile.code}`).then(r => r.ok ? r.json() : null),
        ]).then(([img, prof, cap]) => {
          updateProfile(profile.code, {
            imageDataUri: img?.dataUri,
            cdnUrl: img?.cdnUrl,
            name: prof?.name || profile.username,
            bio: prof?.bio || '',
            caption: cap?.caption || '',
            imageLoading: false,
          });
        });
      }
    }, { rootMargin: '400px' });
    obs.observe(el);
    return () => obs.disconnect();
  }, [profile.code, profile.username, profile.imageDataUri, updateProfile]);

  const generateOpener = async () => {
    updateProfile(profile.code, { openerLoading: true, opener: undefined });
    try {
      const r = await fetch('/api/opener', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code: profile.code,
          username: profile.username,
          name: profile.name || profile.username,
          bio: profile.bio || '',
          caption: profile.caption || '',
        }),
      });
      const data = await r.json();
      updateProfile(profile.code, { opener: data.opener || undefined, openerLoading: false });
    } catch {
      updateProfile(profile.code, { openerLoading: false });
    }
  };

  const [copied, setCopied] = useState(false);
  const copyOpener = () => {
    if (!profile.opener) return;
    navigator.clipboard.writeText(profile.opener);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const displayName = profile.name || profile.full_name || '';
  const scoreClass = profile.score > 0.9 ? 'score-hot' : profile.score > 0.5 ? 'score-warm' : 'score-cold';

  return (
    <div className="card" ref={ref}>
      <div className="card-photo">
        <div className="card-rank">#{rank}</div>
        <div className={`card-score-badge ${scoreClass}`}>{profile.score.toFixed(3)}</div>
        {profile.imageDataUri ? (
          <>
            <a href={profile.link} target="_blank" rel="noreferrer">
              <img src={profile.imageDataUri} alt={profile.username} />
            </a>
            <div className="qr-overlay" title="Scan to open post on phone">
              <QRCodeSVG
                value={profile.link}
                size={120}
                bgColor="transparent"
                fgColor="#ffffff"
                level="M"
              />
            </div>
          </>
        ) : (
          <div className="placeholder">
            {profile.imageLoading ? <span className="spinner" /> : ''}
          </div>
        )}
      </div>

      <div className="card-body">
        <div className="card-header">
          <div className="card-username">
            <a href={`https://www.instagram.com/${profile.username}/`} target="_blank" rel="noreferrer">
              @{profile.username}
            </a>
            {displayName && displayName !== profile.username && (
              <span className="card-name">{displayName}</span>
            )}
          </div>
        </div>

        {profile.caption && <div className="card-caption">{profile.caption}</div>}
        {profile.bio && <div className="card-bio">{profile.bio}</div>}

        {profile.opener ? (
          <div className="opener-box">
            <div className="opener-text">"{profile.opener}"</div>
            <div className="opener-actions">
              <button className="btn btn-copy" onClick={copyOpener}>
                {copied ? 'Copied!' : 'Copy'}
              </button>
              <button className="btn btn-regen" onClick={generateOpener} disabled={profile.openerLoading}>
                {profile.openerLoading ? <><span className="spinner" />...</> : 'Regenerate'}
              </button>
            </div>
          </div>
        ) : (
          <button
            className="btn btn-generate"
            onClick={generateOpener}
            disabled={profile.openerLoading}
          >
            {profile.openerLoading ? <><span className="spinner" />Generating...</> : 'Generate Opener'}
          </button>
        )}
      </div>
    </div>
  );
}
