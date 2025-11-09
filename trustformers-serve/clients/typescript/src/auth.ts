/**
 * TrustformeRS TypeScript Client Authentication
 * 
 * Provides authentication mechanisms for TrustformeRS serving infrastructure.
 * Supports API keys, JWT tokens, OAuth2, and custom authentication methods.
 */

/**
 * Base authentication configuration interface
 */
export interface AuthConfig {
  /**
   * Get authentication headers for requests
   */
  getHeaders(): Record<string, string>;
  
  /**
   * Refresh authentication if supported (e.g., for JWT tokens)
   */
  refresh?(): Promise<void>;
  
  /**
   * Check if authentication is valid/not expired
   */
  isValid?(): boolean;
  
  /**
   * Get authentication type for logging/debugging
   */
  getType(): string;
}

/**
 * API Key authentication
 */
export class APIKeyAuth implements AuthConfig {
  private readonly apiKey: string;
  private readonly headerName: string;
  private readonly prefix: string;

  constructor(
    apiKey: string,
    headerName: string = 'Authorization',
    prefix: string = 'Bearer'
  ) {
    if (!apiKey || apiKey.trim().length === 0) {
      throw new Error('API key cannot be empty');
    }
    
    this.apiKey = apiKey.trim();
    this.headerName = headerName;
    this.prefix = prefix;
  }

  getHeaders(): Record<string, string> {
    const value = this.prefix ? `${this.prefix} ${this.apiKey}` : this.apiKey;
    return {
      [this.headerName]: value,
    };
  }

  isValid(): boolean {
    return this.apiKey.length > 0;
  }

  getType(): string {
    return 'api_key';
  }
}

/**
 * JWT Token authentication
 */
export class JWTAuth implements AuthConfig {
  private token: string;
  private readonly refreshToken?: string;
  private readonly refreshUrl?: string;
  private readonly headerName: string;
  private expiresAt?: Date;
  private refreshPromise?: Promise<void>;

  constructor(
    token: string,
    options: {
      refreshToken?: string;
      refreshUrl?: string;
      headerName?: string;
      expiresAt?: Date;
    } = {}
  ) {
    if (!token || token.trim().length === 0) {
      throw new Error('JWT token cannot be empty');
    }
    
    this.token = token.trim();
    this.refreshToken = options.refreshToken;
    this.refreshUrl = options.refreshUrl;
    this.headerName = options.headerName || 'Authorization';
    this.expiresAt = options.expiresAt;
    
    // Try to extract expiration from token if not provided
    if (!this.expiresAt) {
      this.expiresAt = this.extractExpirationFromToken();
    }
  }

  getHeaders(): Record<string, string> {
    return {
      [this.headerName]: `Bearer ${this.token}`,
    };
  }

  async refresh(): Promise<void> {
    // Prevent multiple concurrent refresh attempts
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    if (!this.refreshToken || !this.refreshUrl) {
      throw new Error('Refresh token and URL required for token refresh');
    }

    this.refreshPromise = this.performRefresh();
    
    try {
      await this.refreshPromise;
    } finally {
      this.refreshPromise = undefined;
    }
  }

  private async performRefresh(): Promise<void> {
    if (!this.refreshUrl || !this.refreshToken) {
      throw new Error('Refresh URL and token required');
    }

    try {
      const response = await fetch(this.refreshUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          refresh_token: this.refreshToken,
        }),
      });

      if (!response.ok) {
        throw new Error(`Token refresh failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      
      if (!data.access_token) {
        throw new Error('No access token in refresh response');
      }

      this.token = data.access_token;
      
      // Update expiration if provided
      if (data.expires_in) {
        this.expiresAt = new Date(Date.now() + data.expires_in * 1000);
      } else if (data.expires_at) {
        this.expiresAt = new Date(data.expires_at * 1000);
      } else {
        // Try to extract from new token
        this.expiresAt = this.extractExpirationFromToken();
      }
    } catch (error) {
      throw new Error(`Token refresh failed: ${error}`);
    }
  }

  isValid(): boolean {
    if (!this.token) {
      return false;
    }
    
    if (this.expiresAt) {
      // Add 5 minute buffer to account for clock skew
      const bufferMs = 5 * 60 * 1000;
      return this.expiresAt.getTime() > Date.now() + bufferMs;
    }
    
    // If no expiration info, assume valid
    return true;
  }

  getType(): string {
    return 'jwt';
  }

  /**
   * Extract expiration time from JWT token payload
   */
  private extractExpirationFromToken(): Date | undefined {
    try {
      const parts = this.token.split('.');
      if (parts.length !== 3) {
        return undefined;
      }
      
      const payload = JSON.parse(atob(parts[1]));
      
      if (payload.exp) {
        return new Date(payload.exp * 1000);
      }
    } catch {
      // Ignore parsing errors
    }
    
    return undefined;
  }

  /**
   * Get token expiration time
   */
  getExpiresAt(): Date | undefined {
    return this.expiresAt;
  }

  /**
   * Check if token will expire within specified seconds
   */
  willExpireWithin(seconds: number): boolean {
    if (!this.expiresAt) {
      return false;
    }
    
    return this.expiresAt.getTime() <= Date.now() + (seconds * 1000);
  }
}

/**
 * OAuth2 authentication with automatic token refresh
 */
export class OAuth2Auth implements AuthConfig {
  private accessToken: string;
  private readonly refreshToken?: string;
  private readonly clientId: string;
  private readonly clientSecret?: string;
  private readonly tokenUrl: string;
  private readonly scopes?: string[];
  private expiresAt?: Date;
  private refreshPromise?: Promise<void>;

  constructor(
    accessToken: string,
    options: {
      refreshToken?: string;
      clientId: string;
      clientSecret?: string;
      tokenUrl: string;
      scopes?: string[];
      expiresAt?: Date;
      expiresIn?: number;
    }
  ) {
    if (!accessToken || accessToken.trim().length === 0) {
      throw new Error('Access token cannot be empty');
    }
    if (!options.clientId || !options.tokenUrl) {
      throw new Error('Client ID and token URL are required');
    }
    
    this.accessToken = accessToken.trim();
    this.refreshToken = options.refreshToken;
    this.clientId = options.clientId;
    this.clientSecret = options.clientSecret;
    this.tokenUrl = options.tokenUrl;
    this.scopes = options.scopes;
    
    // Set expiration time
    if (options.expiresAt) {
      this.expiresAt = options.expiresAt;
    } else if (options.expiresIn) {
      this.expiresAt = new Date(Date.now() + options.expiresIn * 1000);
    }
  }

  getHeaders(): Record<string, string> {
    return {
      'Authorization': `Bearer ${this.accessToken}`,
    };
  }

  async refresh(): Promise<void> {
    // Prevent multiple concurrent refresh attempts
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    if (!this.refreshToken) {
      throw new Error('Refresh token required for OAuth2 token refresh');
    }

    this.refreshPromise = this.performRefresh();
    
    try {
      await this.refreshPromise;
    } finally {
      this.refreshPromise = undefined;
    }
  }

  private async performRefresh(): Promise<void> {
    const body = new URLSearchParams({
      grant_type: 'refresh_token',
      refresh_token: this.refreshToken!,
      client_id: this.clientId,
    });

    if (this.clientSecret) {
      body.append('client_secret', this.clientSecret);
    }

    if (this.scopes && this.scopes.length > 0) {
      body.append('scope', this.scopes.join(' '));
    }

    try {
      const response = await fetch(this.tokenUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: body.toString(),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`OAuth2 token refresh failed: ${response.status} ${errorText}`);
      }

      const data = await response.json();
      
      if (!data.access_token) {
        throw new Error('No access token in OAuth2 refresh response');
      }

      this.accessToken = data.access_token;
      
      // Update expiration
      if (data.expires_in) {
        this.expiresAt = new Date(Date.now() + data.expires_in * 1000);
      }
    } catch (error) {
      throw new Error(`OAuth2 token refresh failed: ${error}`);
    }
  }

  isValid(): boolean {
    if (!this.accessToken) {
      return false;
    }
    
    if (this.expiresAt) {
      // Add 5 minute buffer
      const bufferMs = 5 * 60 * 1000;
      return this.expiresAt.getTime() > Date.now() + bufferMs;
    }
    
    return true;
  }

  getType(): string {
    return 'oauth2';
  }

  getExpiresAt(): Date | undefined {
    return this.expiresAt;
  }

  willExpireWithin(seconds: number): boolean {
    if (!this.expiresAt) {
      return false;
    }
    
    return this.expiresAt.getTime() <= Date.now() + (seconds * 1000);
  }
}

/**
 * Custom header-based authentication
 */
export class CustomHeaderAuth implements AuthConfig {
  private readonly headers: Record<string, string>;

  constructor(headers: Record<string, string>) {
    if (!headers || Object.keys(headers).length === 0) {
      throw new Error('At least one header must be provided');
    }
    
    this.headers = { ...headers };
  }

  getHeaders(): Record<string, string> {
    return { ...this.headers };
  }

  isValid(): boolean {
    return Object.keys(this.headers).length > 0;
  }

  getType(): string {
    return 'custom_header';
  }
}

/**
 * Basic HTTP authentication
 */
export class BasicAuth implements AuthConfig {
  private readonly username: string;
  private readonly password: string;

  constructor(username: string, password: string) {
    if (!username || !password) {
      throw new Error('Username and password are required for basic auth');
    }
    
    this.username = username;
    this.password = password;
  }

  getHeaders(): Record<string, string> {
    const credentials = btoa(`${this.username}:${this.password}`);
    return {
      'Authorization': `Basic ${credentials}`,
    };
  }

  isValid(): boolean {
    return this.username.length > 0 && this.password.length > 0;
  }

  getType(): string {
    return 'basic';
  }
}

/**
 * No authentication (for public endpoints)
 */
export class NoAuth implements AuthConfig {
  getHeaders(): Record<string, string> {
    return {};
  }

  isValid(): boolean {
    return true;
  }

  getType(): string {
    return 'none';
  }
}

/**
 * Automatic authentication wrapper that handles token refresh
 */
export class AutoRefreshAuth implements AuthConfig {
  private auth: AuthConfig;
  private readonly refreshThresholdSeconds: number;

  constructor(auth: AuthConfig, refreshThresholdSeconds: number = 300) {
    this.auth = auth;
    this.refreshThresholdSeconds = refreshThresholdSeconds;
  }

  async getHeaders(): Promise<Record<string, string>> {
    // Check if we need to refresh
    if (this.shouldRefresh()) {
      await this.refresh();
    }
    
    return this.auth.getHeaders();
  }

  getHeaders(): Record<string, string> {
    return this.auth.getHeaders();
  }

  async refresh(): Promise<void> {
    if (this.auth.refresh) {
      await this.auth.refresh();
    }
  }

  isValid(): boolean {
    return this.auth.isValid ? this.auth.isValid() : true;
  }

  getType(): string {
    return `auto_refresh_${this.auth.getType()}`;
  }

  private shouldRefresh(): boolean {
    if (!this.auth.isValid) {
      return false;
    }
    
    // Check if auth supports checking expiration
    if ('willExpireWithin' in this.auth && typeof this.auth.willExpireWithin === 'function') {
      return (this.auth as any).willExpireWithin(this.refreshThresholdSeconds);
    }
    
    return false;
  }
}

// Utility functions

/**
 * Create API key authentication
 */
export function createAPIKeyAuth(
  apiKey: string,
  headerName?: string,
  prefix?: string
): APIKeyAuth {
  return new APIKeyAuth(apiKey, headerName, prefix);
}

/**
 * Create JWT authentication
 */
export function createJWTAuth(
  token: string,
  options?: {
    refreshToken?: string;
    refreshUrl?: string;
    headerName?: string;
    expiresAt?: Date;
  }
): JWTAuth {
  return new JWTAuth(token, options);
}

/**
 * Create OAuth2 authentication
 */
export function createOAuth2Auth(
  accessToken: string,
  options: {
    refreshToken?: string;
    clientId: string;
    clientSecret?: string;
    tokenUrl: string;
    scopes?: string[];
    expiresAt?: Date;
    expiresIn?: number;
  }
): OAuth2Auth {
  return new OAuth2Auth(accessToken, options);
}

/**
 * Create basic HTTP authentication
 */
export function createBasicAuth(username: string, password: string): BasicAuth {
  return new BasicAuth(username, password);
}

/**
 * Create custom header authentication
 */
export function createCustomHeaderAuth(headers: Record<string, string>): CustomHeaderAuth {
  return new CustomHeaderAuth(headers);
}

/**
 * Create auto-refreshing authentication wrapper
 */
export function createAutoRefreshAuth(
  auth: AuthConfig,
  refreshThresholdSeconds?: number
): AutoRefreshAuth {
  return new AutoRefreshAuth(auth, refreshThresholdSeconds);
}