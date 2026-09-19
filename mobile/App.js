import { useState } from 'react';
import { ActivityIndicator, Image, Pressable, SafeAreaView, StatusBar, StyleSheet, Text, View } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';

// Sustituye esta IP por la IP local de la máquina donde corra FastAPI.
const API_URL = 'http://192.168.1.100:8000';

export default function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function chooseImage(source) {
    setError(null);
    setResult(null);
    const options = { mediaTypes: ['images'], quality: 0.9, allowsEditing: true, aspect: [1, 1] };
    const response = source === 'camera'
      ? await ImagePicker.launchCameraAsync(options)
      : await ImagePicker.launchImageLibraryAsync(options);
    if (!response.canceled) setImage(response.assets[0]);
  }

  async function classify() {
    if (!image) return;
    setLoading(true);
    setError(null);
    try {
      const body = new FormData();
      body.append('file', { uri: image.uri, name: 'egg.jpg', type: 'image/jpeg' });
      const response = await fetch(`${API_URL}/predict`, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'No se pudo analizar la imagen.');
      setResult(data);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  const resultIsCrack = result?.label === 'crack';

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" />
      <View style={styles.container}>
        <View style={styles.header}>
          <View style={styles.mark}><Ionicons name="egg" size={24} color="#F4EBDD" /></View>
          <View><Text style={styles.kicker}>CONTROL DE CALIDAD</Text><Text style={styles.title}>Egg Quality</Text></View>
        </View>
        <Text style={styles.intro}>Revisa un huevo en segundos.</Text>
        <Text style={styles.subtitle}>La cámara busca señales visibles de grieta y devuelve una estimación.</Text>

        <View style={styles.preview}>
          {image ? <Image source={{ uri: image.uri }} style={styles.previewImage} /> : (
            <View style={styles.emptyPreview}><Ionicons name="scan-outline" size={48} color="#B56A45" /><Text style={styles.emptyTitle}>Aún no hay una imagen</Text><Text style={styles.emptyHint}>Usa una foto clara y bien iluminada</Text></View>
          )}
        </View>

        <View style={styles.actions}>
          <Pressable style={styles.secondaryButton} onPress={() => chooseImage('camera')}><Ionicons name="camera-outline" size={20} color="#29352D" /><Text style={styles.secondaryText}>Cámara</Text></Pressable>
          <Pressable style={styles.secondaryButton} onPress={() => chooseImage('library')}><Ionicons name="images-outline" size={20} color="#29352D" /><Text style={styles.secondaryText}>Galería</Text></Pressable>
        </View>

        <Pressable disabled={!image || loading} style={[styles.primaryButton, (!image || loading) && styles.disabled]} onPress={classify}>
          {loading ? <ActivityIndicator color="#F4EBDD" /> : <><Ionicons name="sparkles-outline" size={20} color="#F4EBDD" /><Text style={styles.primaryText}>Analizar huevo</Text></>}
        </Pressable>

        {error && <View style={styles.error}><Ionicons name="alert-circle-outline" size={20} color="#9E3F35" /><Text style={styles.errorText}>{error}</Text></View>}
        {result && <View style={[styles.result, resultIsCrack ? styles.crackResult : styles.goodResult]}><Ionicons name={resultIsCrack ? 'warning-outline' : 'checkmark-circle-outline'} size={30} color={resultIsCrack ? '#9E3F35' : '#3E654C'} /><View><Text style={styles.resultLabel}>{resultIsCrack ? 'Posible grieta' : 'Aparenta estar bueno'}</Text><Text style={styles.resultMeta}>Confianza: {(result.confidence * 100).toFixed(1)}%</Text></View></View>}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F4EBDD' },
  container: { flex: 1, padding: 24, paddingTop: 32 },
  header: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  mark: { width: 48, height: 48, borderRadius: 16, backgroundColor: '#B56A45', alignItems: 'center', justifyContent: 'center' },
  kicker: { color: '#B56A45', fontSize: 11, fontWeight: '800', letterSpacing: 1.5 },
  title: { color: '#29352D', fontSize: 25, fontWeight: '800' },
  intro: { marginTop: 38, color: '#29352D', fontSize: 32, fontWeight: '800', letterSpacing: -0.5 },
  subtitle: { marginTop: 8, color: '#6E756D', fontSize: 16, lineHeight: 23 },
  preview: { marginTop: 26, height: 265, borderRadius: 24, overflow: 'hidden', backgroundColor: '#E8D9C7', borderWidth: 1, borderColor: '#D7C5B1' },
  previewImage: { width: '100%', height: '100%' },
  emptyPreview: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  emptyTitle: { marginTop: 12, color: '#29352D', fontSize: 17, fontWeight: '700' },
  emptyHint: { marginTop: 5, color: '#7C817A', fontSize: 13 },
  actions: { flexDirection: 'row', gap: 12, marginTop: 16 },
  secondaryButton: { flex: 1, height: 48, borderRadius: 14, backgroundColor: '#FFF9F0', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8 },
  secondaryText: { color: '#29352D', fontWeight: '700', fontSize: 15 },
  primaryButton: { height: 54, borderRadius: 16, marginTop: 14, backgroundColor: '#29352D', alignItems: 'center', justifyContent: 'center', flexDirection: 'row', gap: 9 },
  primaryText: { color: '#F4EBDD', fontSize: 16, fontWeight: '800' },
  disabled: { opacity: 0.45 },
  error: { marginTop: 14, padding: 14, borderRadius: 14, backgroundColor: '#F6D8D2', flexDirection: 'row', gap: 8, alignItems: 'center' },
  errorText: { color: '#9E3F35', flex: 1 },
  result: { marginTop: 14, padding: 18, borderRadius: 18, flexDirection: 'row', gap: 12, alignItems: 'center' },
  goodResult: { backgroundColor: '#DCE9DD' },
  crackResult: { backgroundColor: '#F6D8D2' },
  resultLabel: { color: '#29352D', fontSize: 18, fontWeight: '800' },
  resultMeta: { marginTop: 3, color: '#6E756D', fontSize: 14 }
});
