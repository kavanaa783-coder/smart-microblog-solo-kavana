import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 15000,
});

export const getCurrentUsername = () => localStorage.getItem('username') || 'mahimashree';

export const scanPost = async (content) => {
  const { data } = await api.post('/scan-post', { text: content });
  return normalizeScanResult(data);
};

export const savePost = async ({ content, scan }) => {
  const username = getCurrentUsername();
  const { data } = await api.post('/save-post', {
    username,
    content,
    risk_score: scan?.riskScore,
    risk_level: scan?.riskLevel,
  });
  return normalizePost(data);
};

export const getFeed = async () => {
  const { data } = await api.get('/get-feed');
  const list = Array.isArray(data) ? data : data?.posts || [];
  return list.map(normalizePost);
};

export const deletePost = async (id) => {
  await api.delete(`/delete-post/${id}`);
  return true;
};
export const createReply = async (postId, content) => {
  const username = getCurrentUsername();

  const { data } = await api.post(`/post/${postId}/replies`, {
    username,
    content,
  });

  return data;
};

export const getReplies = async (postId) => {
  const { data } = await api.get(`/post/${postId}/replies`);
  return data?.replies || [];
};

export const getPost = async (id) => {
  const { data } = await api.get(`/post/${id}`);
  return normalizePost(data);
};

export const updatePost = async (id, content) => {
  const { data } = await api.put(`/update-post/${id}`, { content });
  return normalizePost(data);
};

export const getStats = async () => {
  const { data } = await api.get('/stats');
  return normalizeStats(data);
};

export const getProfile = async (username = getCurrentUsername()) => {
  const { data } = await api.get(`/profile/${username}`);
  return normalizeProfile(data);
};

export const saveProfile = async ({ username = getCurrentUsername(), bio, profileImage = '' }) => {
  localStorage.setItem('username', username);
  const { data } = await api.post('/profile', {
    username,
    bio,
    profile_image: profileImage,
  });
  return normalizeProfile(data);
};

function normalizeScanResult(data = {}) {
  return {
    entities: data.detected_entities || {},
    riskScore: data.risk_score ?? 0,
    riskLevel: (data.risk_level || 'LOW').toUpperCase(),
    recommendation: data.recommendation || '',
  };
}

function normalizePost(data = {}) {
  return {
    id: data.id,
    content: data.content || '',
    username: data.username || 'mahimashree',
    handle: '@' + (data.username || 'mahimashree').toLowerCase(),
    riskScore: data.risk_score ?? 0,
    riskLevel: (data.risk_level || 'LOW').toUpperCase(),
    entities: data.detected_entities || {},
    createdAt: data.timestamp || new Date().toISOString(),
  };
}

function normalizeStats(data = {}) {
  return {
    total: data.total_posts ?? 0,
    low: data.low_risk ?? 0,
    medium: data.medium_risk ?? 0,
    high: data.high_risk ?? 0,
  };
}

function normalizeProfile(data = {}) {
  return {
    id: data.id,
    username: data.username || '',
    bio: data.bio || '',
    profileImage: data.profile_image || '',
    joinedDate: data.joined_date || null,
  };
}

export default api;