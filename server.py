#!/usr/bin/env python3
"""
ORACLE Chatbot Backend Server
Python Flask server with ORACLE agent overseer integration

Usage:
    python server.py

The server will start on http://localhost:3000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
import json
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['JSON_SORT_KEYS'] = False
PORT = int(os.environ.get('PORT', 3000))
DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'


class Oracle:
    """
    ORACLE Agent Overseer
    Manages approval of agent actions based on risk assessment
    """

    def __init__(self):
        self.approved = 0
        self.denied = 0
        self.saved = 0.0
        self.mind_reading = True
        self.history = []

    def check(self, agent: str, action: str, cost: float = 0, gpu_min: int = 0, thought: str = "") -> Dict[str, Any]:
        """
        Check if an action is approved by ORACLE

        Args:
            agent: Name of the agent
            action: Action description
            cost: Estimated cost in USD
            gpu_min: Minimum GPU memory required (MB)
            thought: Agent's reasoning

        Returns:
            Dict with approval status and reason
        """
        # Risk assessment
        risk_factors = [
            cost > 10,
            gpu_min > 100,
            'delete' in action.lower(),
            '24/7' in action.lower()
        ]
        risk = 'high' if any(risk_factors) else 'low'
        approved = risk == 'low'

        # Update stats
        if approved:
            self.approved += 1
        else:
            self.denied += 1
            self.saved += cost

        # Log to history
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent': agent,
            'action': action,
            'cost': cost,
            'gpu_min': gpu_min,
            'risk': risk,
            'approved': approved,
            'thought': thought
        }
        self.history.append(log_entry)

        # Reason
        if approved:
            reason = f"✅ ORACLE APPROVED: Low risk operation"
        else:
            reasons = []
            if cost > 10:
                reasons.append(f"cost exceeds threshold (${cost} > $10)")
            if gpu_min > 100:
                reasons.append(f"GPU usage too high ({gpu_min}MB > 100MB)")
            if 'delete' in action.lower():
                reasons.append("action contains 'delete'")
            if '24/7' in action.lower():
                reasons.append("action contains '24/7'")
            reason = f"⛔ ORACLE DENIED: {', '.join(reasons)}"

        logger.info(f"{reason} | Agent: {agent}")

        return {
            'approved': approved,
            'risk': risk,
            'reason': reason,
            'saved': cost if not approved else 0
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get ORACLE statistics"""
        return {
            'approved': self.approved,
            'denied': self.denied,
            'saved': round(self.saved, 2),
            'mind_reading': self.mind_reading
        }

    def get_history(self, limit: int = 50) -> list:
        """Get action history"""
        return self.history[-limit:]


# Initialize ORACLE
oracle = Oracle()


# ==================== ROUTES ====================

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'ORACLE Chatbot Backend'
    }), 200


@app.route('/api/chat', methods=['POST'])
def send_message():
    """
    Send a message to the chatbot

    Request Body:
        {
            "content": "What actions are running?",
            "conversationId": "conv-123"
        }

    Response:
        {
            "id": "msg-1234567890",
            "content": "Bot response here",
            "sender": "bot",
            "timestamp": "2026-07-27T10:30:00Z",
            "agentAction": {...}
        }
    """
    try:
        data = request.get_json()

        # Validate request
        if not data:
            return jsonify({
                'message': 'Request body is required',
                'error': 'VALIDATION_ERROR'
            }), 400

        content = data.get('content', '').strip()
        conversation_id = data.get('conversationId', '')

        if not content:
            return jsonify({
                'message': 'Message content is required',
                'error': 'VALIDATION_ERROR'
            }), 400

        # Generate bot response
        message_id = f'msg-{int(datetime.now().timestamp() * 1000)}'
        timestamp = datetime.now().isoformat()

        # Simple echo response for MVP
        # TODO: Integrate with real LLM (OpenAI, Claude, Ollama, etc.)
        bot_response = f'You said: "{content}". I\'m ready to help with ORACLE operations!'

        # Create agent action for demonstration
        agent_action = {
            'agent': 'chatbot',
            'action': 'process_message',
            'risk': 'low',
            'cost': 0,
            'approved': True
        }

        response = {
            'id': message_id,
            'content': bot_response,
            'sender': 'bot',
            'timestamp': timestamp,
            'agentAction': agent_action
        }

        logger.info(f"Message processed: {conversation_id} - {content[:50]}...")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return jsonify({
            'message': 'Failed to process message',
            'error': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/oracle/check', methods=['POST'])
def check_action():
    """
    Check if an agent action is approved by ORACLE

    Request Body:
        {
            "agent": "vision_agent",
            "action": "train YOLO 70B",
            "cost": 12.5,
            "gpu_min": 240,
            "thought": "overkill for 20 photos"
        }

    Response:
        {
            "approved": false,
            "risk": "high",
            "reason": "⛔ ORACLE DENIED: ...",
            "saved": 12.5
        }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'message': 'Request body is required',
                'error': 'VALIDATION_ERROR'
            }), 400

        agent = data.get('agent', '').strip()
        action = data.get('action', '').strip()
        cost = float(data.get('cost', 0))
        gpu_min = int(data.get('gpu_min', 0))
        thought = data.get('thought', '').strip()

        if not agent or not action:
            return jsonify({
                'message': 'Agent and action are required',
                'error': 'VALIDATION_ERROR'
            }), 400

        # Check with ORACLE
        result = oracle.check(agent, action, cost, gpu_min, thought)
        return jsonify(result), 200

    except ValueError as e:
        logger.error(f"Invalid value in request: {str(e)}")
        return jsonify({
            'message': 'Invalid parameter values',
            'error': 'VALIDATION_ERROR'
        }), 400
    except Exception as e:
        logger.error(f"Error checking action: {str(e)}")
        return jsonify({
            'message': 'Failed to check action',
            'error': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/oracle/stats', methods=['GET'])
def get_oracle_stats():
    """
    Get ORACLE statistics

    Response:
        {
            "approved": 42,
            "denied": 8,
            "saved": 95.50,
            "mind_reading": true
        }
    """
    try:
        stats = oracle.get_stats()
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({
            'message': 'Failed to get stats',
            'error': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/oracle/history', methods=['GET'])
def get_oracle_history():
    """
    Get ORACLE action history

    Query Parameters:
        limit: Number of records to return (default: 50, max: 200)

    Response:
        {
            "history": [...],
            "count": 42
        }
    """
    try:
        limit = min(int(request.args.get('limit', 50)), 200)
        history = oracle.get_history(limit)
        return jsonify({
            'history': history,
            'count': len(history)
        }), 200
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        return jsonify({
            'message': 'Failed to get history',
            'error': 'INTERNAL_ERROR'
        }), 500


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'message': 'Endpoint not found',
        'error': 'NOT_FOUND'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'message': 'Method not allowed',
        'error': 'METHOD_NOT_ALLOWED'
    }), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'message': 'Internal server error',
        'error': 'INTERNAL_ERROR'
    }), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    logger.info(f"🚀 Starting ORACLE Chatbot Backend...")
    logger.info(f"📡 Server: http://localhost:{PORT}")
    logger.info(f"🔍 Debug mode: {DEBUG}")
    logger.info(f"💡 Endpoints:")
    logger.info(f"   GET  /health")
    logger.info(f"   POST /api/chat")
    logger.info(f"   POST /api/oracle/check")
    logger.info(f"   GET  /api/oracle/stats")
    logger.info(f"   GET  /api/oracle/history")

    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
