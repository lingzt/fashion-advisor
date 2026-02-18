"use client";

import { useState, useEffect, useRef } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface ImageResult {
  image: string;
  match_score: number;
  match_level: string;
  match_reasons: string;
}

interface Recommendation {
  images: ImageResult[];
  advice: string;
  debug?: {
    parsed_by_llm?: {
      identity?: string;
      occasion?: string;
      style?: string;
      color?: string;
      gender?: string;
      special_needs?: string;
    };
  };
}

const CORRECT_PASSWORD = "ling2024"; // Change this password
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001";

export default function FashionAISearch() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [password, setPassword] = useState("");
  const [passwordError, setPasswordError] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [images, setImages] = useState<ImageResult[]>([]);
  const [advice, setAdvice] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Check if already authenticated
    const auth = sessionStorage.getItem("fashion_auth");
    if (auth === "true") {
      setIsAuthenticated(true);
    }
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  const handlePasswordSubmit = () => {
    if (password === CORRECT_PASSWORD) {
      sessionStorage.setItem("fashion_auth", "true");
      setIsAuthenticated(true);
      setPasswordError(false);
    } else {
      setPasswordError(true);
      setPassword("");
    }
  };

  const handleLogout = () => {
    sessionStorage.removeItem("fashion_auth");
    setIsAuthenticated(false);
    setMessages([]);
    setImages([]);
    setPassword("");
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput("");
    setLoading(true);

    setMessages(prev => [...prev, { role: "user", content: userMessage }]);

    try {
      const response = await fetch(`${API_URL}/api/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ identity: userMessage }),
      });

      if (!response.ok) throw new Error("API request failed");

      const data: Recommendation = await response.json();
      setImages(data.images || []);
      setAdvice(data.advice || "");

      if (data.debug?.parsed_by_llm) {
        const parsed = data.debug.parsed_by_llm;
        const parts: string[] = [];
        if (parsed.identity) parts.push(`Query: ${parsed.identity}`);
        if (parsed.gender) parts.push(`👤 ${parsed.gender}`);
        if (parsed.occasion) parts.push(`🎯 ${parsed.occasion}`);
        if (parsed.color) parts.push(`🎨 ${parsed.color}`);
        if (parsed.style) parts.push(`✨ ${parsed.style}`);
        if (parsed.special_needs) parts.push(`💡 ${parsed.special_needs}`);
        
        setMessages(prev => [...prev, {
          role: "assistant",
          content: `📝 **Parsed Request:**\n${parts.join(" | ")}`
        }]);
      }

      setMessages(prev => [...prev, {
        role: "assistant",
        content: data.advice || "Here are some recommendations based on your request!"
      }]);
    } catch (error) {
      console.error("Error:", error);
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "❌ Sorry, I encountered an error. Please try again."
      }]);
    }

    setLoading(false);
  };

  // Password Screen
  if (!isAuthenticated) {
    return (
      <div style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
      }}>
        <div style={{
          background: "white",
          padding: "40px",
          borderRadius: "16px",
          boxShadow: "0 10px 40px rgba(0,0,0,0.2)",
          textAlign: "center",
          maxWidth: "400px",
          width: "90%"
        }}>
          <h1 style={{ marginBottom: "10px", color: "#333" }}>🎨 Fashion AI</h1>
          <p style={{ marginBottom: "30px", color: "#666" }}>Enter password to access</p>
          
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handlePasswordSubmit()}
            placeholder="Enter password..."
            style={{
              width: "100%",
              padding: "15px",
              border: passwordError ? "2px solid #dc3545" : "2px solid #ddd",
              borderRadius: "8px",
              fontSize: "16px",
              marginBottom: "15px",
              outline: "none",
              boxSizing: "border-box"
            }}
          />
          
          {passwordError && (
            <p style={{ color: "#dc3545", marginBottom: "15px" }}>❌ Incorrect password</p>
          )}
          
          <button
            onClick={handlePasswordSubmit}
            style={{
              width: "100%",
              padding: "15px",
              background: "#667eea",
              color: "white",
              border: "none",
              borderRadius: "8px",
              fontSize: "16px",
              cursor: "pointer"
            }}
          >
            Enter
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto", padding: "20px", fontFamily: "system-ui" }}>
      {/* Logout Button */}
      <div style={{ textAlign: "right", marginBottom: "10px" }}>
        <button
          onClick={handleLogout}
          style={{
            background: "transparent",
            border: "1px solid #ddd",
            padding: "5px 15px",
            borderRadius: "5px",
            cursor: "pointer",
            fontSize: "12px",
            color: "#666"
          }}
        >
          🔒 Logout
        </button>
      </div>

      <h1 style={{ textAlign: "center", color: "#333", marginBottom: "30px" }}>
        🎨 Fashion AI Search
      </h1>

      {/* Chat Messages */}
      <div style={{
        border: "1px solid #ddd",
        borderRadius: "12px",
        padding: "20px",
        marginBottom: "20px",
        minHeight: "300px",
        maxHeight: "400px",
        overflowY: "auto",
        background: "#fafafa"
      }}>
        {messages.length === 0 && (
          <p style={{ color: "#888", textAlign: "center" }}>
            Ask me for fashion recommendations! Try: &quot;red dress for a wedding&quot;
          </p>
        )}
        {messages.map((msg, i) => (
          <div key={i} style={{
            marginBottom: "15px",
            textAlign: msg.role === "user" ? "right" : "left"
          }}>
            <div style={{
              display: "inline-block",
              padding: "10px 15px",
              borderRadius: "15px",
              background: msg.role === "user" ? "#007bff" : "#fff",
              color: msg.role === "user" ? "#fff" : "#333",
              border: msg.role === "assistant" ? "1px solid #ddd" : "none",
              maxWidth: "80%",
              whiteSpace: "pre-wrap"
            }}>
              {msg.content}
            </div>
          </div>
        ))}
        {loading && (
          <p style={{ color: "#888", textAlign: "center" }}>🔍 Searching...</p>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Image Results */}
      {images.length > 0 && (
        <div style={{ marginBottom: "20px" }}>
          <h3 style={{ marginBottom: "15px" }}>📸 {images.length} Results Found</h3>
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))",
            gap: "15px"
          }}>
            {images.map((img, i) => (
              <div key={i} style={{
                border: "1px solid #eee",
                borderRadius: "8px",
                overflow: "hidden",
                textAlign: "center"
              }}>
                <img
                  src={`${API_URL}/${img.image}`}
                  alt={`Result ${i + 1}`}
                  style={{ width: "100%", height: "150px", objectFit: "cover" }}
                />
                <div style={{ padding: "8px", fontSize: "12px" }}>
                  <div style={{
                    background: img.match_level === "✅" ? "#28a745" :
                               img.match_level === "🟡" ? "#ffc107" : "#6c757d",
                    color: "#fff",
                    padding: "2px 8px",
                    borderRadius: "10px",
                    display: "inline-block",
                    marginBottom: "5px"
                  }}>
                    {img.match_level} {img.match_score}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div style={{ display: "flex", gap: "10px" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Describe what you&apos;re looking for..."
          style={{
            flex: 1,
            padding: "15px",
            border: "1px solid #ddd",
            borderRadius: "25px",
            fontSize: "16px",
            outline: "none"
          }}
        />
        <button
          onClick={handleSend}
          disabled={loading}
          style={{
            padding: "15px 30px",
            background: loading ? "#ccc" : "#007bff",
            color: "#fff",
            border: "none",
            borderRadius: "25px",
            cursor: loading ? "not-allowed" : "pointer",
            fontSize: "16px"
          }}
        >
          {loading ? "..." : "Search"}
        </button>
      </div>
    </div>
  );
}
