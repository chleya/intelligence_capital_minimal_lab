import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def prepare_counterfactual_data(df, seed, env_kwargs, device="cpu"):
    """Extract features and targets from a counterfactual table split."""
    import json

    obs_dim = env_kwargs["state_dim"]
    history_len = env_kwargs["history_len"]
    feature_dim = history_len * (obs_dim + 1)

    X_full = []
    Y_full = []
    for _, row in df.iterrows():
        hist_obs = row.get("history_obs")
        hist_act = row.get("history_act")
        if hist_obs is None or hist_act is None:
            continue
        if isinstance(hist_obs, str):
            hist_obs = json.loads(hist_obs)
        if isinstance(hist_act, str):
            hist_act = json.loads(hist_act)
        parts = []
        for o, a in zip(hist_obs, hist_act):
            oa = np.array(o, dtype=np.float32)
            parts.append(np.concatenate([oa, [float(a)]]))
        x = np.concatenate(parts).astype(np.float32)

        def parse(v):
            if isinstance(v, str):
                return np.array(json.loads(v), dtype=np.float32)
            return np.array(v, dtype=np.float32)

        y_m1 = float(np.sum(parse(row["outcome_m1"])))
        y_0 = float(np.sum(parse(row["outcome_0"])))
        y_p1 = float(np.sum(parse(row["outcome_p1"])))
        X_full.append(x)
        Y_full.append([y_m1, y_0, y_p1])

    if len(X_full) == 0:
        return None, None, None

    X_full = np.stack(X_full)
    Y_full = np.array(Y_full, dtype=np.float32)
    best_action = np.argmax(Y_full, axis=1).astype(np.int64)
    return X_full, Y_full, best_action


def train_state_only_classifier(model, X_train, Y_train, X_val, Y_val,
                                epochs=200, batch_size=64, lr=0.001, weight_decay=1e-5,
                                patience=20, device="cpu"):
    model = model.to(device)
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(np.argmax(Y_train, axis=1), dtype=torch.long).to(device)

    if X_val is not None:
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_v = torch.tensor(np.argmax(Y_val, axis=1), dtype=torch.long).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience // 2)

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_v)
                val_loss = nn.functional.cross_entropy(val_logits, y_v).item()
            scheduler.step(val_loss)
            current = val_loss
        else:
            current = total_loss / len(loader)

        if current < best_loss:
            best_loss = current
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience * 2:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_ae_model(model, X_train, Y_train, X_val, Y_val, model_type,
                   epochs=400, batch_size=64, lr=0.001, weight_decay=1e-5,
                   patience=40, device="cpu", ce_weight=0.8):
    """Hybrid training: per-(state,action) MSE + periodic predict_all_actions CE."""
    model = model.to(device)
    n_states = len(X_train)

    X_expanded, y_expanded, a_expanded = [], [], []
    for i in range(n_states):
        for a_val in range(3):
            X_expanded.append(X_train[i])
            y_expanded.append(Y_train[i, a_val])
            a_expanded.append(a_val)
    X_te = torch.tensor(np.stack(X_expanded), dtype=torch.float32).to(device)
    y_te = torch.tensor(np.array(y_expanded), dtype=torch.float32).unsqueeze(1).to(device)
    a_te = torch.tensor(np.array(a_expanded), dtype=torch.long).to(device)

    if X_val is not None:
        nv = len(X_val)
        Xv_e, yv_e, av_e = [], [], []
        for i in range(nv):
            for a_val in range(3):
                Xv_e.append(X_val[i])
                yv_e.append(Y_val[i, a_val])
                av_e.append(a_val)
        X_ve = torch.tensor(np.stack(Xv_e), dtype=torch.float32).to(device)
        y_ve = torch.tensor(np.array(yv_e), dtype=torch.float32).unsqueeze(1).to(device)
        a_ve = torch.tensor(np.array(av_e), dtype=torch.long).to(device)

    dataset_ae = TensorDataset(X_te, y_te, a_te)
    loader_ae = DataLoader(dataset_ae, batch_size=batch_size * 3, shuffle=True)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    ba_t = torch.tensor(np.argmax(Y_train, axis=1), dtype=torch.long).to(device)

    if X_val is not None:
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        Y_v = torch.tensor(Y_val, dtype=torch.float32).to(device)
        ba_v = torch.tensor(np.argmax(Y_val, axis=1), dtype=torch.long).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience // 2)

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_mse = 0.0

        for xb, yb, ab in loader_ae:
            optimizer.zero_grad()
            if model_type == "residual":
                y_hat, b_hat, r_hat = model(xb, ab)
                loss = nn.functional.mse_loss(y_hat, yb)
            elif model_type == "centered_residual":
                y_hat, m_hat, cr_hat = model(xb, ab)
                loss = nn.functional.mse_loss(y_hat, yb)
            elif model_type == "residual_adversarial":
                y_hat, b_hat, r_hat, adv = model(xb, ab)
                mse_loss = nn.functional.mse_loss(y_hat, yb)
                adv_loss = nn.functional.mse_loss(adv, torch.zeros_like(adv))
                loss = mse_loss + 0.1 * adv_loss
            else:
                y_hat = model(xb, ab)
                loss = nn.functional.mse_loss(y_hat, yb)

            loss.backward()
            optimizer.step()
            total_mse += loss.item()

        if hasattr(model, 'predict_all_actions') and ce_weight > 0:
            optimizer.zero_grad()
            all_preds = model.predict_all_actions(X_t)
            ce_loss = ce_weight * nn.functional.cross_entropy(all_preds, ba_t)
            ce_loss.backward()
            optimizer.step()

        if X_val is not None and hasattr(model, 'predict_all_actions'):
            model.eval()
            with torch.no_grad():
                all_val = model.predict_all_actions(X_v)
                val_mse = nn.functional.mse_loss(all_val, Y_v).item()
                val_ce = nn.functional.cross_entropy(all_val, ba_v).item()
                val_loss = val_mse + ce_weight * val_ce
            scheduler.step(val_loss)
            current = val_loss
        elif X_val is not None:
            model.eval()
            with torch.no_grad():
                if model_type == "residual_adversarial":
                    val_y, _, _, _ = model(X_ve, a_ve)
                elif model_type in ("residual", "centered_residual"):
                    val_y, _, _ = model(X_ve, a_ve)
                else:
                    val_y = model(X_ve, a_ve)
                val_loss = nn.functional.mse_loss(val_y, y_ve).item()
            scheduler.step(val_loss)
            current = val_loss
        else:
            current = total_mse / len(loader_ae)

        if current < best_loss:
            best_loss = current
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_counterfactual_joint(model, X_train, Y_train, X_val, Y_val,
                               epochs=500, batch_size=64, lr=0.001, weight_decay=1e-5,
                               patience=40, device="cpu", ce_weight=1.0, normalize_y=True):
    """Train CounterfactualCompressor: input h, output 3 values jointly.
    normalize_y=True standardizes outcomes for better CE stability."""
    model = model.to(device)

    y_mean = np.mean(Y_train, axis=0, keepdims=True)
    y_std = np.std(Y_train, axis=0, keepdims=True)
    y_std = np.maximum(y_std, 1e-6)
    if normalize_y:
        Y_tr_norm = (Y_train - y_mean) / y_std
        Y_val_norm = (Y_val - y_mean) / y_std if Y_val is not None else None
    else:
        Y_tr_norm = Y_train
        Y_val_norm = Y_val

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(Y_tr_norm, dtype=torch.float32).to(device)
    ba_t = torch.tensor(np.argmax(Y_train, axis=1), dtype=torch.long).to(device)

    if X_val is not None:
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_v = torch.tensor(Y_val_norm, dtype=torch.float32).to(device)
        ba_v = torch.tensor(np.argmax(Y_val, axis=1), dtype=torch.long).to(device)

    dataset = TensorDataset(X_t, y_t, ba_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience // 2)

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb, bab in loader:
            optimizer.zero_grad()
            y_hat = model(xb)
            mse = nn.functional.mse_loss(y_hat, yb)
            ce = nn.functional.cross_entropy(y_hat, bab)
            loss = 0.1 * mse + ce_weight * ce
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_y = model(X_v)
                val_mse = nn.functional.mse_loss(val_y, y_v).item()
                val_ce = nn.functional.cross_entropy(val_y, ba_v).item()
                val_loss = 0.1 * val_mse + ce_weight * val_ce
            scheduler.step(val_loss)
            current = val_loss
        else:
            current = total_loss / len(loader)

        if current < best_loss:
            best_loss = current
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_causal_contrast(model, X_train, Y_train, X_val, Y_val,
                          epochs=200, batch_size=64, lr=0.001, weight_decay=1e-5,
                          patience=20, device="cpu"):
    model = model.to(device)
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    best_labels = torch.tensor(np.argmax(Y_train, axis=1), dtype=torch.long).to(device)

    if X_val is not None:
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_v = torch.tensor(Y_val, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience // 2)

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            y_hat, z = model(xb)
            mse_loss = nn.functional.mse_loss(y_hat, yb)
            idx_in_batch = torch.randint(0, len(best_labels), (len(xb),), device=device)
            c_loss = model.contrastive_loss(z, best_labels[idx_in_batch])
            loss = mse_loss + 0.1 * c_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_y, _ = model(X_v)
                val_loss = nn.functional.mse_loss(val_y, y_v).item()
            scheduler.step(val_loss)
            current = val_loss
        else:
            current = total_loss / len(loader)

        if current < best_loss:
            best_loss = current
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience * 2:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_memory_mechanism(mechanism, X_train, outcomes_train):
    mechanism.fit(X_train, [outcomes_train[:, 0], outcomes_train[:, 1], outcomes_train[:, 2]])
    return mechanism